import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

import bot
import config
import db_reset
import strategies_100
import strategy_engine as se
import telegram_views as views
import tick_candles
import trading_sessions as sessions


def update_for(data=None, text=None):
    update = Mock()
    update.effective_chat.id = 7
    update.effective_user.id = 1
    update.effective_message.text = text
    update.callback_query.data = data
    update.callback_query.message.message_id = 55
    return update


class SessionTests(unittest.TestCase):
    def test_dst_session_boundaries_and_mismatched_change_weeks(self):
        cases = {
            "2026-06-01": {"06:59": "Asia", "07:00": "London", "11:59": "London", "12:00": "Overlap", "14:59": "Overlap", "15:00": "NY", "19:59": "NY", "20:00": "Other"},
            "2026-01-02": {"06:59": "Asia", "07:00": "Other", "08:00": "London", "12:59": "London", "13:00": "Overlap", "15:59": "Overlap", "16:00": "NY", "20:59": "NY", "21:00": "Other"},
        }
        for day, times in cases.items():
            for clock, expected in times.items():
                value = datetime.fromisoformat(day + "T" + clock)
                self.assertEqual(sessions.session_name(value), expected)
                self.assertEqual(sessions.session_name(value.replace(tzinfo=timezone.utc).astimezone(sessions.KYIV)), expected)

    def test_daily_report_schedule_and_period_start(self):
        self.assertEqual(sessions.next_daily_report(datetime.fromisoformat("2026-03-28T18:00")), datetime.fromisoformat("2026-03-28T19:00").replace(tzinfo=timezone.utc))
        self.assertEqual(sessions.period_start("today", datetime(2026, 6, 1, 22)), datetime(2026, 6, 1, 21))
        self.assertIsNone(sessions.period_start("all", datetime(2026, 1, 1)))


class ViewTests(unittest.TestCase):
    def test_wr_tags_card_and_binary_points(self):
        self.assertEqual(views.format_wr_tag({"wr": 62.4, "total": 18, "source": "LIVE"}), "[L:62%·18]")
        self.assertEqual(views.format_wr_tag({"wr": 41.0, "total": 50, "source": "BT"}), "[B:41%·50]")
        self.assertEqual(views.format_wr_tag(None), "[—]")

        cand = dict(strategy="REV_RSI14", signal="CALL", rsi=12, bar_end=datetime(2026, 6, 1, 14))
        card = views.signal_card("EUR/USD", cand, 1.2, datetime(2026, 6, 1, 14))
        self.assertNotIn("REV_RSI14", card)
        for v in ["🟢 CALL", "ЕКСПІРАЦІЯ: 60 ХВИЛИН", "18:00 Київ", "Впевненість:", "Overlap (Лондон + Нью-Йорк)"]:
            self.assertIn(v, card)

        s_jpy_loss = views.settlement_text("LOSS", 159.60950, -14, pair="USD/JPY", entry_price=159.595, signal_type="PUT")
        self.assertIn("159.610", s_jpy_loss)
        self.assertIn("-14.5 пт", s_jpy_loss)


class Strategy157AndLifecycleTests(unittest.TestCase):
    def test_registry_157_unique_constraints_and_64_byte_callbacks(self):
        se.sync_registry_from_deleted(set())
        self.assertEqual(len(se.FULL_STRATEGIES_META), 157)
        self.assertEqual(len(strategies_100.EXTENDED_META), 100)
        self.assertEqual(len(se.ALL_STRATEGY_KEYS), 157)
        for key, meta in se.FULL_STRATEGIES_META.items():
            self.assertLessEqual(len(meta["title"]), 22, key)
            self.assertIn(meta["category"], se.STRATEGY_CATEGORIES)
            self.assertTrue(1 <= meta["exp"] <= 60)
            for pair in config.PAIRS_MAP:
                for cb in (f"pair_toggle|{pair}|{key}|0", f"autodis_en|P|{pair}|{key}", f"del_one_ok|{key}"):
                    self.assertLessEqual(len(cb.encode("utf-8")), 64, cb)

    def test_all_100_new_strategies_strictly_causal(self):
        rng = np.random.default_rng(777)
        idx = pd.date_range("2026-11-10 00:00", periods=500, freq="5min")
        idx = idx[(idx.weekday != 5) & ~((idx.weekday == 6) & (idx.hour < 22))]
        c = 1.1000 + np.cumsum(rng.standard_t(df=4, size=len(idx)) * 0.0003)
        o = np.roll(c, 1); o[0] = c[0]
        h = np.maximum(o, c) + np.abs(rng.normal(0, 0.00025, len(idx)))
        l = np.minimum(o, c) - np.abs(rng.normal(0, 0.00025, len(idx)))
        v = rng.lognormal(2.5, 0.7, len(idx)).astype(float)
        df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)

        ind_f = se.compute_base_indicators(df)
        ind_p = se.compute_base_indicators(df.iloc[:260])
        htf_f = {"trend": np.zeros(len(df), dtype=np.int8), "rsi": np.full(len(df), 50.0)}
        htf_p = {"trend": np.zeros(260, dtype=np.int8), "rsi": np.full(260, 50.0)}

        for key in strategies_100.EXTENDED_META:
            s_full = se.evaluate_strategy_series(df, key, ind=ind_f, htf_ctx=htf_f, regime_filter=False)
            s_part = se.evaluate_strategy_series(df.iloc[:260], key, ind=ind_p, htf_ctx=htf_p, regime_filter=False)
            np.testing.assert_array_equal(s_full[:260], s_part, f"Look-ahead in {key}")

    def test_delete_and_restore_strategies_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            cfg_p = tmp / "strategies_config.json"
            del_p = tmp / "deleted_strategies.json"
            cfg_p.write_text(json.dumps({"MR_Z50_5M": True, "TR_HOLY_5M": True}), encoding="utf-8")
            se.sync_registry_from_deleted(set())
            try:
                removed = se.delete_strategies_permanently(["MR_Z50_5M"], config_path=cfg_p, deleted_path=del_p)
                self.assertEqual(removed, ["MR_Z50_5M"])
                self.assertNotIn("MR_Z50_5M", se.STRATEGIES_META)
                self.assertNotIn("MR_Z50_5M", json.loads(cfg_p.read_text()))
                restored = se.restore_all_deleted_strategies(config_path=cfg_p, deleted_path=del_p)
                self.assertEqual(restored, 1)
                self.assertIn("MR_Z50_5M", se.STRATEGIES_META)
                self.assertTrue(json.loads(cfg_p.read_text())["MR_Z50_5M"])
            finally:
                se.sync_registry_from_deleted(set())


class CallbackAndTelegramControlTests(unittest.TestCase):
    def test_all_telegram_controls_and_winrate_buttons(self):
        se.sync_registry_from_deleted(set())
        db = Mock()
        db.get_pair_strategy_status.return_value = True
        db.get_chat_signals.return_value = []
        db.get_auto_disabled_map.return_value = {}
        db.get_auto_disabled_items.return_value = []
        db.get_detailed_pair_strategy_stats.return_value = []
        db.get_backtest_results.return_value = []
        db.get_all_bot_settings.return_value = {}
        db.get_strategy_winrate_summary.return_value = {"MR_VWAP_Z_5M": {"wr": 64.5, "total": 31, "source": "BT"}}
        db.clear_all_stats.return_value = True
        db.delete_strategies_data.return_value = True

        cfg = dict.fromkeys(se.ALL_STRATEGY_KEYS, True) | {"session_filter": False, "regime_filter": True, "news_filter": True, "auto_disable_49": True}
        with patch.multiple(bot, database=db, strategies_status=cfg, ADMIN_IDS={1}, save_strategies_config=Mock(), AUTO_SCAN_CHATS=set()), \
             patch.object(bot.threading, "Thread"), \
             patch.object(bot.db_reset, "reset_database_and_config", return_value={"tables": ["signals"], "config_reset": True}):
            mk_mean = bot.strategy_keyboard(category="MEAN", page=0, wr_summary=db.get_strategy_winrate_summary())
            btn_texts = [b.text for row in mk_mean.inline_keyboard for b in row]
            self.assertTrue(any("[B:64%·31]" in t or "[B:65%·31]" in t for t in btn_texts))

            markups = [
                bot.inline_main_keyboard(7), bot.stats_keyboard(), bot.clear_stats_confirm_keyboard(),
                bot.db_reset_confirm_keyboard(), bot.active_keyboard(), bot.scan_keyboard(),
                bot.check_pairs_keyboard(), bot.backtest_keyboard("all"), bot.settings_menu_keyboard(),
                bot.lab_settings_keyboard(), bot.thresholds_keyboard(), bot.confluence_keyboard(),
                bot.expiry_keyboard(), bot.filters_keyboard(), bot.tz_keyboard(),
                bot.strategy_keyboard(), bot.delete_strategies_keyboard(wr_summary=db.get_strategy_winrate_summary()),
            ]
            for cat in se.STRATEGY_CATEGORIES:
                markups.append(bot.strategy_keyboard(category=cat, page=0))
            callbacks = {btn.callback_data for mk in markups for row in mk.inline_keyboard for btn in row}
            for data in callbacks:
                self.assertLessEqual(len(data.encode("utf-8")), 64, data)
                u = update_for(data)
                bot.handle_callback(u, None)
                u.callback_query.answer.assert_called_once()


class SecurityAndModulesTests(unittest.TestCase):
    def test_tick_candles_and_db_reset(self):
        sym = "TEST_T_157"
        tick_candles._bars.pop(sym, None)
        tick_candles.add_tick(sym, 1.10, 6000000)
        tick_candles.add_tick(sym, 1.12, 6120000)
        tick_candles.add_tick(sym, 1.11, 6060000)
        self.assertEqual(list(tick_candles._bars[sym].keys()), [100, 101, 102])
        self.assertTrue(config._sanitize_database_url("postgres://u:p@host/db?channel_binding=require&sslmode=require").startswith("postgresql://"))

        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__.return_value.fetchall.return_value = [("signals",)]
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = Path(tmpdir) / "strategies_config.json"
            with patch.object(db_reset.database, "get_connection", return_value=mock_conn):
                res = db_reset.reset_database_and_config(cfg_file, reset_config=True)
            self.assertEqual(res["tables"], ["signals"])
            self.assertEqual(len(json.loads(cfg_file.read_text())), 161)


if __name__ == "__main__":
    unittest.main()
