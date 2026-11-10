import ast
import html
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, KeyboardButton, ReplyKeyboardMarkup

import ai_advisor
import backtest
import database
import finnhub_ws
import indicators
import ml_model
import tick_candles
from signal_pipeline import build_closed_frames, closed_only
from timeframe_profiles import PROFILES


ROOT = Path(__file__).parent
BOT_TREE = ast.parse((ROOT / "bot.py").read_text())


def bot_functions():
    namespace = dict(pd=pd, datetime=datetime, timedelta=timedelta, timezone=timezone,
                     html=html, os=os, json=json, logger=logging.getLogger("batch1"),
                     InlineKeyboardButton=InlineKeyboardButton, InlineKeyboardMarkup=InlineKeyboardMarkup,
                     KeyboardButton=KeyboardButton, ReplyKeyboardMarkup=ReplyKeyboardMarkup)
    functions = [node for node in BOT_TREE.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(body=functions, type_ignores=[]), "bot_functions", "exec"), namespace)
    return namespace


def primary():
    return pd.DataFrame(dict(open=1., close=1.001, high=1.002, low=1., atr=.001,
                             ema_20=1., vwap=1.002, rsi=55., bb_lower=.9, bb_upper=1.1,
                             wick_ratio=.4, stoch_k=90., macd_hist=-1.),
                        index=pd.date_range("2026-01-01", periods=60, freq="min"))


class EngineTests(unittest.TestCase):
    def test_all_switches_off_and_legacy_fallback(self):
        a = indicators.AdaptiveTechnicalAnalysis()
        p = primary()
        with patch.object(database, "get_pair_strategy_status", return_value=True) as status, patch.object(indicators, "detect_market_regime", return_value="RANGE_BOUND"):
            result = a.generate_master_confluence_signal(p, p, p, p, p, "NEUTRAL", {}, strategies_status=dict(trend_multi=False, flat_corridor=False, price_action=False, oscillators=False))
            self.assertEqual(result["signal"], "NONE")
            status.assert_not_called()
            a.generate_master_confluence_signal(p, p, p, p, p, "NEUTRAL", {}, strategies_status=dict(trend_multi=False, flat_corridor=False, market_levels=False, volume_volatility=False))
            status.assert_not_called()

    def test_direction_specific_attribution(self):
        a = indicators.AdaptiveTechnicalAnalysis()
        p = primary()
        local = pd.DataFrame({"close": np.linspace(.98, 1.02, 60)})
        with patch.object(database, "get_pair_strategy_status", return_value=True), patch.object(indicators, "detect_market_regime", return_value="RANGE_BOUND"), patch.object(a, "get_min_dist_to_pivot_or_round", return_value=.0001):
            r = a.generate_master_confluence_signal(p, local, p, local, local, "BULLISH", {})
        self.assertEqual((r["signal"], r["strategy"]), ("PUT", "PRICE_LEVELS"))
        self.assertNotIn("BULLISH", r["reason"])
        self.assertIn("PUT", r["reason"])

    def test_sweep_uses_directional_wick_and_flat_bars(self):
        a = indicators.AdaptiveTechnicalAnalysis()
        for direction in ("CALL", "PUT"):
            for correct in (True, False):
                p = primary()
                values = dict(open=1., close=1.01 if direction == "CALL" else .99,
                              high=1.02, low=.98)
                values["low" if (direction == "CALL") == correct else "high"] = .9 if (direction == "CALL") == correct else 1.1
                for key, value in values.items():
                    p.loc[p.index[-1], key] = value
                with patch.object(database, "get_pair_strategy_status", return_value=True), patch.object(indicators, "detect_market_regime", return_value="CHAOS"), patch.object(indicators, "calculate_dynamic_threshold", return_value=2.):
                    r = a.generate_master_confluence_signal(p, p, p, p, p, "NEUTRAL", {}, strategies_status=dict(trend_multi=False, flat_corridor=False, price_action=True, oscillators=False))
                self.assertEqual(r["signal"], direction if correct else "NONE")
        p[["open", "close", "high", "low"]] = 1.
        with patch.object(database, "get_pair_strategy_status", return_value=True), patch.object(indicators, "detect_market_regime", return_value="CHAOS"):
            self.assertEqual(a.generate_master_confluence_signal(p,p,p,p,p,"NEUTRAL",{})["signal"], "NONE")


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
             expires_at=self.now - timedelta(seconds=late), chat_id=7, message_text="")
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


class AITests(unittest.TestCase):
    def test_fail_modes_missing_sdk_invalid_and_empty_decisions(self):
        for mode in ("open", "closed"):
            for response in (None, "not JSON", "{}", '{"decision":"YES"}', '{"decision":"YES","confidence":99}'):
                advisor = ai_advisor.AIAdvisor()
                advisor.api_key = "fake"
                advisor.fail_mode = mode
                model = SimpleNamespace(generate_content=lambda *a, **k: SimpleNamespace(text=response))
                with patch.object(ai_advisor, "HAS_GENAI", response is not None), patch.object(ai_advisor, "genai", SimpleNamespace(GenerativeModel=lambda _:model)), patch.object(advisor,"get_available_models",return_value=["fake"]):
                    result = advisor.evaluate_signal("EUR/USD", {"suggested_exp":3},None,None,None)
                self.assertEqual(result["decision"], "YES" if mode == "open" else "NO")
                self.assertFalse(result["ai_available"])

    def test_prompt_labels_expiry_and_valid_decision(self):
        advisor = ai_advisor.AIAdvisor()
        advisor.api_key = "fake"
        generate = Mock(return_value=SimpleNamespace(text='{"decision":"YES","confidence":8,"suggested_expiration":60}'))
        with patch.object(ai_advisor,"HAS_GENAI",True), patch.object(ai_advisor,"genai",SimpleNamespace(GenerativeModel=lambda _:SimpleNamespace(generate_content=generate))), patch.object(advisor,"get_available_models",return_value=["fake"]):
            result = advisor.evaluate_signal("EUR/USD", dict(tf="3m", suggested_exp=3, chart_tfs=("1h","5m","3m"),exp_range=(3,6)),None,None,None)
        self.assertTrue(result["ai_available"])
        self.assertEqual(result["suggested_expiration"],6)
        text = " ".join(generate.call_args.args[0])
        self.assertIn("5m",text)
        self.assertIn("RSI (3m)",text)
        self.assertIn("від 3 до 6",text)
        self.assertNotIn("1M",text)


class BotFlowTests(unittest.TestCase):
    def setUp(self):
        self.ns = bot_functions()
        self.now = datetime(2026,1,2,12,5)
        self.cand = dict(tf="3m",pair="EUR/USD",signal="CALL",strategy="TREND_IMPULSE",current_price=1.2,
                         atr=.002,bar_end=self.now-timedelta(seconds=10),suggested_exp=3,exp_range=(3,6),
                         confirmations=2,confirmation_names=[],ml_min_auto=53,ml_min_manual=44)
        self.gate = Mock()
        self.gate.check.return_value = (True,"OK")
        self.db = Mock()
        self.db.save_signal.return_value = 123
        ml = Mock(is_active=False)
        ml.predict_proba.return_value = 55
        self.advisor = Mock()
        self.advisor.evaluate_signal.return_value = dict(decision="YES",confidence=7,suggested_expiration=3,ai_available=False)
        self.ns.update(PROFILES=PROFILES, validate_signal_conditions=lambda _: (True,"OK"),
            get_current_session_info=lambda: ("session",1,12),segment_filter=Mock(check=lambda *a:(True,"OK")),
            signal_gate=self.gate, ml_filter=ml, ML_FILTER_ENABLED=True,check_pivot_level_proximity=lambda _: (True,"OK"),
            ai_advisor_instance=self.advisor, get_utc_now=lambda:self.now, get_live_price=lambda _:1.2,
            database=self.db, bot=Mock(), schedule_signal_timer=Mock(), build_closed_frames=build_closed_frames,
            create_chart_image=Mock(return_value=None), ADMIN_IDS=set(), AUTO_SCAN_CHATS=set())

    def process(self, frames=None):
        return self.ns["process_candidate"](7,"EUR/USD","TEST",self.cand,frames or {},True,None,False)

    def test_normal_save_records_expiry_and_registers(self):
        with patch.dict(os.environ,{"GEMINI_API_KEY":"","GEMINI_KEY":""}):
            self.assertTrue(self.process()[0])
        self.assertEqual(self.db.save_signal.call_args.kwargs["expires_at"],self.now+timedelta(minutes=3))
        self.assertEqual(self.gate.check.call_count,2)
        self.gate.register.assert_called_once()

    def test_failed_save_marks_message_without_registering(self):
        self.db.save_signal.return_value=None
        self.assertFalse(self.process()[0])
        self.gate.register.assert_not_called()
        self.ns["schedule_signal_timer"].assert_not_called()
        self.assertIn("не збережено",self.ns["bot"].send_message.call_args.kwargs["text"])

    def test_post_ai_freshness_gate_and_adverse_price(self):
        self.cand["bar_end"]=self.now-timedelta(seconds=181)
        self.assertFalse(self.process()[0])
        self.cand["bar_end"]=self.now-timedelta(seconds=10)
        self.gate.check.side_effect=[(True,"OK"),(False,"duplicate")]
        self.assertEqual(self.process(),(False,"duplicate"))
        self.gate.check.side_effect=None
        for direction, price in (("CALL",1.198),("PUT",1.202)):
            self.cand["signal"]=direction
            self.ns["get_live_price"]=lambda _:price
            self.assertIn("0.5 ATR",self.process()[1])
        self.gate.register.assert_not_called()

    def test_charts_are_closed_and_actual_labels_passed(self):
        df=primary()
        df.index=pd.date_range(self.now-timedelta(minutes=59),periods=60,freq="min")
        with patch.dict(os.environ,{"GEMINI_API_KEY":"fake"}):
            self.process({"1h":df,"5m":df,"3m":df})
        payload=self.advisor.evaluate_signal.call_args.args[1]
        self.assertEqual(payload["chart_tfs"],("1h","5m","3m"))
        self.assertEqual(payload["exp_range"],(3,6))
        for call in self.ns["create_chart_image"].call_args_list:
            chart, label=call.args
            minutes={"1h":60,"5m":5,"3m":3}[call.kwargs["tf_label"]]
            self.assertTrue(((chart.index+pd.Timedelta(minutes=minutes)) <= self.now).all())

    def test_admin_access_and_legacy_settings(self):
        update=SimpleNamespace(effective_user=SimpleNamespace(id=2),effective_message=Mock())
        self.ns["ADMIN_IDS"]={1}
        self.assertFalse(self.ns["require_admin"](update))
        update.effective_user.id=1
        self.assertTrue(self.ns["require_admin"](update))
        self.ns["ADMIN_IDS"]=set()
        self.assertTrue(self.ns["require_admin"](update))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"strategies.json"
            path.write_text(json.dumps(dict(market_levels=False,volume_volatility=False)))
            self.ns.update(STRATEGIES_CONFIG_FILE=str(path),GROUPED_STRATEGIES=dict(price_action="p",oscillators="o"))
            self.assertEqual(self.ns["load_strategies_config"]()["price_action"],False)
            path.write_text(json.dumps(dict(price_action=True,market_levels=False)))
            self.assertTrue(self.ns["load_strategies_config"]()["price_action"])

    def test_dashboard_handlers_empty_and_aggregate(self):
        self.ns.update(PAIRS_MAP={"EUR/USD":"TEST"})
        for data in ("dash_pairs","dash_strategies"):
            for rows in ([],[dict(ticker="TEST",strategy="X",wins=3,total=5),dict(ticker="TEST",strategy="X",wins=1,total=5)]):
                self.db.get_detailed_pair_strategy_stats.return_value=rows
                query=Mock(data=data)
                query.message.chat_id=7
                self.ns["handle_callback"](SimpleNamespace(callback_query=query),None)
                text=query.edit_message_text.call_args.args[0]
                self.assertIn("40.0%" if rows else "Недостатньо",text)

    def test_main_menu_uses_reply_keyboard_and_guarded_actions(self):
        self.assertIsInstance(self.ns["main_keyboard"](7),ReplyKeyboardMarkup)
        query=Mock(data="back_to_main_menu")
        query.message.chat_id=7
        self.ns["handle_callback"](SimpleNamespace(callback_query=query),None)
        self.assertIsInstance(query.message.reply_text.call_args.kwargs["reply_markup"],ReplyKeyboardMarkup)
        self.ns["ADMIN_IDS"]={1}
        for data in ("toggle_group_price_action","toggle_cpair_EUR/USD_price_action","bt_EUR/USD","retrain_ml","clear_stats_do"):
            query=Mock(data=data)
            query.message.chat_id=7
            update=SimpleNamespace(callback_query=query,effective_user=SimpleNamespace(id=2),effective_message=Mock())
            self.ns["handle_callback"](update,None)
            update.effective_message.reply_text.assert_called_once()
            query.edit_message_text.assert_not_called()


class BacktestAndPersistenceTests(unittest.TestCase):
    def test_macro_load_warmup_and_missing_history(self):
        import yfinance
        hourly=pd.DataFrame({"Open":1.,"High":2.,"Low":.9,"Close":1.2,"Volume":1},index=pd.date_range("2026-01-01",periods=300,freq="h",tz="UTC"))
        daily=hourly.resample("1D").last()
        with patch.object(yfinance,"download",side_effect=[hourly,daily]) as download:
            h,d=backtest.load_macro("EUR/USD")
            self.assertEqual(len(h),300)
            self.assertIsNone(h.index.tz)
            self.assertEqual(download.call_args_list[0].kwargs["period"],"60d")
        with patch.object(yfinance,"download",side_effect=[hourly.iloc[:10],daily]):
            with self.assertRaisesRegex(ValueError,"260"):
                backtest.load_macro("EUR/USD")
        with patch.object(yfinance,"download",side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(RuntimeError,"offline"):
                backtest.load_macro("EUR/USD")

    def test_chronological_events_exact_opens_and_warmup(self):
        start=pd.Timestamp("2026-01-15")
        index=pd.date_range(start,periods=800,freq="min")
        df=pd.DataFrame({"open":np.arange(800)+1.,"close":9000.,"high":10000.,"low":0.,"volume":1},index=index)
        hourly=pd.DataFrame({"open":1.,"close":np.linspace(1,2,400),"high":2.,"low":.9,"volume":1},index=pd.date_range(end=start,periods=400,freq="h"))
        daily=backtest._resample(hourly,"1D")
        times=[]
        def generate(pair,frames,now,status,profiles):
            times.append(now)
            self.assertGreaterEqual(len(closed_only(frames["1h"],"1h",now)),260)
            self.assertGreaterEqual(len(frames["1d"]),2)
            self.assertEqual(status,{"price_action":False})
            return [dict(pair=pair,tf=p,signal="CALL",tf_minutes=PROFILES[p]["minutes"],bar_id=f"{p}|{now}",suggested_exp=3,confirmations=2,current_price=9000.) for p in profiles],{}
        with patch.object(backtest,"generate_candidates",side_effect=generate),patch.object(backtest,"validate_signal_conditions",return_value=(True,"OK")),patch.object(backtest,"check_pivot_level_proximity",return_value=(True,"OK")):
            trades=backtest.simulate_pair("EUR/USD",df,step_cap=2,strategies_status={"price_action":False},df_1h=hourly,df_daily=daily)
        self.assertEqual(times,sorted(times))
        self.assertTrue(trades)
        self.assertTrue(all(t["win"] for t in trades))
        self.assertEqual(backtest.simulate_pair("EUR/USD",df.iloc[:10],strategies_status={}),[])

    def test_ml_rejected_model_stays_disabled_after_reload(self):
        frame=pd.DataFrame({key:np.arange(80)%2 for key in ml_model.FEATURE_NAMES})
        frame["divergence"]="NONE"
        frame["result"]=["WIN","LOSS"]*40
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/"model.pkl")
            model=ml_model.TradingMLFilter(path)
            model.model="previous model"
            model.save_model()
            estimator=Mock()
            estimator.predict_proba.return_value=np.tile([.5,.5],(20,1))
            with patch.object(ml_model,"_new_estimator",return_value=estimator):
                success,_=model.train_from_frame(frame)
            self.assertFalse(success)
            self.assertFalse(ml_model.TradingMLFilter(path).is_active)

    def test_midnight_pivots_use_previous_day(self):
        index=pd.date_range("2026-01-14 20:00",periods=300,freq="min")
        df=pd.DataFrame({"open":1.,"close":1.,"high":1.1,"low":.9,"volume":1},index=index)
        daily=pd.DataFrame({"open":[1.,2.],"close":[1.,2.],"high":[1.1,2.1],"low":[.9,1.9],"volume":1},index=pd.to_datetime(["2026-01-13","2026-01-14"]))
        seen=[]
        def generate(pair,frames,now,status,profiles):
            if now == pd.Timestamp("2026-01-15"):
                seen.append(indicators.AdaptiveTechnicalAnalysis().calculate_pivots(frames["1d"])["P"])
            return [],{}
        with patch.object(backtest,"generate_candidates",side_effect=generate):
            backtest.simulate_pair("EUR/USD",df,profiles=["3m"],strategies_status={},df_daily=daily)
        self.assertEqual(seen,[2.])

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
