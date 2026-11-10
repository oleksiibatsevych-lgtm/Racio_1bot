import ast
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import bot
import database
import finnhub_ws
import tick_candles

ROOT = Path(__file__).parent
BOT_TREE = ast.parse((ROOT / "bot.py").read_text())


def bot_functions():
    namespace = dict(vars(bot))
    namespace["ai_advisor_instance"] = Mock(api_key=None)
    functions = [node for node in BOT_TREE.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), "bot_functions", "exec"), namespace)
    return namespace


class QuoteAndSettlementTests(unittest.TestCase):
    def setUp(self):
        self.ns = bot_functions()
        self.now = datetime(2026, 1, 2, 12, 5, 10)
        self.ns.update(get_utc_now=lambda: self.now, get_live_quote=Mock(return_value=None),
                       tick_candles=SimpleNamespace(get_1m=Mock(return_value=pd.DataFrame())),
                       normalize_yahoo_ticker=lambda *args: "EURUSD=X", fetch_yahoo_fallback=Mock(return_value=pd.DataFrame()))

    def test_ws_quote_preserves_price_api(self):
        with patch.dict(finnhub_ws.live_prices, {}, clear=True), patch.object(tick_candles, "add_tick"):
            finnhub_ws.on_message(None, json.dumps({"type":"trade", "data":[{"s":"TEST", "p":1.2,"t":1700000000000}]}))
            self.assertEqual(finnhub_ws.get_live_quote("TEST"), (1.2, 1700000000.))
            self.assertEqual(finnhub_ws.get_live_price("TEST"), 1.2)
            self.assertIsNone(finnhub_ws.get_live_quote("missing"))

    def test_exit_quote_must_be_fresh_and_near_expiry(self):
        stamp = self.now.replace(tzinfo=timezone.utc).timestamp()
        self.ns["get_live_quote"].return_value = (1.2, stamp - 5)
        self.assertEqual(self.ns["get_exit_price"]("TEST", self.now - timedelta(seconds=10)), 1.2)
        self.assertIsNone(self.ns["get_exit_price"]("TEST", self.now - timedelta(minutes=5)))
        self.ns["get_live_quote"].return_value = (1.2, stamp - 30)
        self.assertIsNone(self.ns["get_exit_price"]("TEST", self.now - timedelta(seconds=30)))

    def test_exact_expiry_candle_only_and_no_flat_fill(self):
        expiry = self.now - timedelta(minutes=4)
        minute = pd.Timestamp(expiry).floor("min")
        ticks = pd.DataFrame({"close":[1.3], "volume":[1]}, index=[minute])
        self.ns["tick_candles"].get_1m.return_value = ticks
        self.assertEqual(self.ns["get_exit_price"]("TEST", expiry), 1.3)
        ticks["volume"] = 0
        self.assertIsNone(self.ns["get_exit_price"]("TEST", expiry))
        self.ns["fetch_yahoo_fallback"].return_value = pd.DataFrame({"close":[1.4]}, index=[minute])
        self.assertEqual(self.ns["get_exit_price"]("TEST", expiry), 1.4)
        self.ns["fetch_yahoo_fallback"].return_value.index += pd.Timedelta(minutes=1)
        self.assertIsNone(self.ns["get_exit_price"]("TEST", expiry))

    def settlement(self, late, price):
        db = Mock()
        db.get_signal_by_id.return_value = dict(id=1, status="PENDING", ticker="TEST", entry_price=1.2,
             signal_type="CALL", expiration_mins=3, timestamp_str="2026-01-02 11:00:00",
             expires_at=self.now - timedelta(seconds=late), chat_id=7, message_id=9, message_text="")
        db.claim_signal.return_value = True
        db.get_stats.return_value = {"wins":0,"losses":0}
        self.ns.update(database=db, bot=Mock(), threading=Mock(), get_exit_price=Mock(return_value=price),
                       PAIRS_MAP={}, ml_filter=Mock())
        self.ns["process_signal_expiration"](1)
        return db

    def test_late_missing_quote_voids(self):
        db = self.settlement(181, None)
        db.update_signal_result.assert_called_once_with(1, "NEUTRAL", None, 0, status="VOID")
        db.release_signal.assert_not_called()

    def test_missing_quote_retries_and_valid_quote_wins(self):
        db = self.settlement(20, None)
        db.release_signal.assert_called_once_with(1)
        db.update_signal_result.assert_not_called()
        db = self.settlement(5, 1.3)
        self.assertEqual(db.update_signal_result.call_args.args[1], "WIN")

    def test_no_early_settlement(self):
        db = self.settlement(-5, 1.3)
        db.claim_signal.assert_not_called()



class PersistenceTests(unittest.TestCase):
    def test_database_expiry_insert_and_void_exclusion(self):
        connection=Mock()
        cursor=connection.cursor.return_value
        cursor.fetchone.return_value=(123,)
        expiry=datetime(2026,1,1)
        with patch.object(database,"get_connection",return_value=connection):
            self.assertEqual(database.save_signal(expires_at=expiry),123)
            sql,args=cursor.execute.call_args.args
            self.assertEqual(sql.count("%s"),len(args))
            self.assertEqual(args[-1],expiry)
            for name in ("get_stats","get_detailed_pair_strategy_stats","get_segment_stats","get_strategy_time_stats"):
                getattr(database,name)()
                self.assertIn("VOID",cursor.execute.call_args.args[0])



if __name__ == "__main__":
    unittest.main()
