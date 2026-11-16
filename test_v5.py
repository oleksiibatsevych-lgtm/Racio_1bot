import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

import ai_advisor
import backtest
import bot
import config
import database
import reversion_engine as rev
import strategy_engine as se
import telegram_views as views
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
            "2026-03-10": {"07:00": "Other", "08:00": "London", "12:00": "Overlap", "16:00": "NY", "20:00": "Other"},
            "2026-10-27": {"07:00": "Other", "08:00": "London", "12:00": "Overlap", "16:00": "NY", "20:00": "Other"},
        }
        for day, times in cases.items():
            for clock, expected in times.items():
                value = datetime.fromisoformat(day + "T" + clock)
                self.assertEqual(sessions.session_name(value), expected, (day, clock))
                self.assertEqual(sessions.session_name(value.replace(tzinfo=timezone.utc).astimezone(sessions.KYIV)), expected)

    def test_session_masks_match_vectorized_research_definition(self):
        for start in ["2026-03-01", "2026-10-01"]:
            idx = pd.date_range(start, periods=24 * 60 * 35, freq="min", tz="UTC")
            london = idx.tz_convert("Europe/London").hour
            ny = idx.tz_convert("America/New_York").hour
            l = (london >= 8) & (london < 16)
            n = (ny >= 8) & (ny < 16)
            expected = np.select([l & n, l & ~n, n & ~l, idx.hour < 7], ["Overlap", "London", "NY", "Asia"], default="Other")
            self.assertEqual([sessions.session_name(t) for t in idx], list(expected))

    def test_selection_uses_bar_end_and_falls_through_disallowed_preset(self):
        df = pd.DataFrame(dict(open=1.0, high=1.1, low=0.9, close=1.0, volume=1), index=pd.date_range("2026-01-01 16:20", periods=201, freq="5min"))
        ind = rev.calculate_indicators(df)
        ind.loc[ind.index[-1], list(rev.PRESETS)] = 1
        now = ind.index[-1] + timedelta(minutes=5)
        self.assertEqual(sessions.session_name(now), "London")
        cfg = dict.fromkeys(rev.PRESETS, True) | {"REV_RSI14": False, "session_filter": True}
        self.assertEqual(rev.candidates_from_indicators("EUR/USD", ind, now, cfg)[0]["strategy"], "REV_RSI7")
        cfg["session_filter"] = False
        self.assertEqual(rev.candidates_from_indicators("EUR/USD", ind, now, cfg)[0]["strategy"], "REV_BB3")

    def test_daily_report_schedule_dst_exact_time_and_period_start(self):
        for now, expected in [("2026-03-28T18:00", "2026-03-28T19:00"), ("2026-03-28T20:00", "2026-03-29T18:00"), ("2026-10-24T20:00", "2026-10-25T19:00"), ("2026-01-01T19:00", "2026-01-02T19:00")]:
            self.assertEqual(sessions.next_daily_report(datetime.fromisoformat(now)), datetime.fromisoformat(expected).replace(tzinfo=timezone.utc))
        self.assertEqual(sessions.period_start("today", datetime(2026, 6, 1, 22)), datetime(2026, 6, 1, 21))
        self.assertIsNone(sessions.period_start("all", datetime(2026, 1, 1)))
        with self.assertRaises(ValueError):
            sessions.period_start("invalid", datetime(2026, 1, 1))


class ViewTests(unittest.TestCase):
    def test_human_stats_strips_oanda_and_long_bars(self):
        rows = [
            dict(status="CLOSED", result="WIN", ticker="OANDA:AUD_USD", strategy="REV_RSI14"),
            dict(status="CLOSED", result="LOSS", ticker="OANDA:AUD_USD", strategy="REV_RSI14"),
            dict(status="CLOSED", result="NEUTRAL", ticker="OANDA:EUR_USD", strategy="REV_RSI14"),
            dict(status="VOID", result="WIN", ticker="OANDA:EUR_USD", strategy="REV_RSI14"),
        ]
        text = views.stats_text(rows, "all")
        self.assertIn("50.0%", text)
        self.assertIn("1✅ / 1❌", text)
        self.assertIn("AUD/USD", text)
        self.assertNotIn("OANDA:", text)
        self.assertNotIn("█████░░░░░", text)
        self.assertIn("Угод за цей період немає", views.stats_text([], "today"))
        for value in [-100, 0, 50, 100, 200]:
            self.assertEqual(len(views.progress_bar(value)), 10)

    def test_matrix_radar_and_autodis_views(self):
        live = [dict(ticker="OANDA:EUR_USD", strategy="DIV_RSI_5M", wins=6, total=10)]
        bt = [dict(pair_name="EUR/USD", strategy_key="STARC_5M", wins=18, total=30)]
        mtxt = views.stats_matrix_text(live, bt)
        self.assertIn("EUR/USD", mtxt)
        self.assertNotIn("OANDA:", mtxt)
        self.assertIn("DIV_RSI_5M", mtxt)
        self.assertIn("STARC_5M", mtxt)
        self.assertIn("Жодна стратегія зараз не заблокована", views.auto_disabled_text([]))

        diag = dict(pair="EUR/USD", session="Overlap", chop=51.0, adx=19.2, rsi14=42.5, vol_ratio=1.1, trend_15m="🟢 Висхідний", trend_60m="🟡 Боковий", checked=40, off_global=10, off_pair=5, off_session=2, muted=0)
        radar = views.pair_radar_text(diag, (False, "OK"), [], live, bt, live_price=1.12345)
        self.assertIn("EUR/USD", radar)
        self.assertIn("1.12345", radar)
        self.assertIn("STARC_5M", radar)

    def test_card_times_stats_session_and_result_views(self):
        cand = dict(strategy="REV_RSI14", signal="CALL", rsi=12, bar_end=datetime(2026, 6, 1, 14))
        card = views.signal_card("EUR/USD", cand, 1.2, datetime(2026, 6, 1, 14))
        for value in ["REV_RSI14", "🟢 CALL", "60 хв", "15:00 UTC / 18:00 Київ", "67.03% · n=276", "Overlap"]:
            self.assertIn(value, card)
        cand_conc = dict(strategy="REV_RSI14", tf="5m", signal="CALL", rsi=22.0, bar_end=datetime(2026, 6, 1, 14), exp_mode="DYN", suggested_exp=24, confluence_score=2.55, call_votes=3, put_votes=1, agreeing_strategies=["REV_RSI14", "STARC_5M", "DIV_RSI_5M"])
        card_conc = views.signal_card("OANDA:GBP_USD", cand_conc, 1.3, datetime(2026, 6, 1, 14), risk_amount=20.0)
        self.assertIn("GBP/USD", card_conc)
        self.assertNotIn("OANDA:", card_conc)
        self.assertIn("Золота середина", card_conc)
        self.assertIn("Консенсус", card_conc)
        self.assertIn("$20.00", card_conc)
        self.assertIn("VOID", views.settlement_text("VOID", None, 0))
        self.assertIn("Відкритих угод немає", views.active_text([], datetime(2026, 1, 1)))
        row = dict(id=1, ticker="OANDA:EUR_USD", strategy="REV_RSI7", signal_type="PUT", expiration_mins=45, created_at=datetime(2026, 1, 1), expires_at=datetime(2026, 1, 1, 0, 1), status="CLOSED", result="WIN", exit_price=1.2, pips=10, entry_price=1.2010)
        act = views.active_text([row], datetime(2026, 1, 1), live_prices={"OANDA:EUR_USD": 1.2000})
        self.assertIn("EUR/USD", act)
        self.assertNotIn("OANDA:", act)
        self.assertIn("+10.0 п.", act)
        self.assertIn("WIN", views.history_text([row]))
        self.assertIn("Завершених угод немає", views.history_text([]))

    def test_query_chat_scope_period_history_limit_and_connection_cleanup(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []
        with patch.object(database, "get_connection", return_value=connection):
            database.get_chat_signals(7, "history", since=datetime(2026, 1, 1))
            sql, params = cursor.execute.call_args.args
            self.assertIn("chat_id = %s", sql)
            self.assertIn("LIMIT 15", sql)
            self.assertEqual(params, [7, datetime(2026, 1, 1)])
            database.get_chat_signals(7, "active")
            self.assertIn("status = 'PENDING'", cursor.execute.call_args.args[0])
            cursor.execute.side_effect = RuntimeError("DB unavailable")
            with self.assertRaises(RuntimeError):
                database.get_chat_signals(7)
        self.assertEqual(connection.close.call_count, 3)
        with self.assertRaises(ValueError):
            database.get_chat_signals(7, "invalid")


class StrategyEngineTests(unittest.TestCase):
    def test_registry_constraints_and_callback_sizes(self):
        self.assertEqual(len(se.ALL_STRATEGY_KEYS), 57)
        for key, meta in se.STRATEGIES_META.items():
            self.assertLessEqual(len(key), 50)
            self.assertIn(meta["category"], se.STRATEGY_CATEGORIES)
            self.assertTrue(1 <= meta["exp"] <= 60)
            for pair in config.PAIRS_MAP:
                cb = f"pair_toggle|{pair}|{key}"
                self.assertLessEqual(len(cb.encode("utf-8")), 64, cb)

    def test_confluence_and_golden_middle_dynamic_expiration(self):
        cands = [
            {"strategy": "REV_RSI14", "signal": "CALL", "suggested_exp": 60, "tf_minutes": 5, "weight": 1.4, "vol_ratio": 1.0, "priority": 100},
            {"strategy": "STARC_5M", "signal": "CALL", "suggested_exp": 15, "tf_minutes": 5, "weight": 1.15, "vol_ratio": 1.0, "priority": 86},
            {"strategy": "DIV_RSI_5M", "signal": "CALL", "suggested_exp": 15, "tf_minutes": 5, "weight": 1.20, "vol_ratio": 1.0, "priority": 91},
            {"strategy": "EMA_RIB_3M", "signal": "PUT", "suggested_exp": 9, "tf_minutes": 3, "weight": 0.95, "vol_ratio": 1.0, "priority": 77},
            {"strategy": "PA_PIN_BB_5M", "signal": "PUT", "suggested_exp": 15, "tf_minutes": 5, "weight": 1.10, "vol_ratio": 1.0, "priority": 84},
        ]
        res, reason = se.resolve_confluence(cands, conflict_mode="CONC", threshold=1.0, expiry_mode="DYN")
        self.assertEqual(reason, "OK")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["signal"], "CALL")
        self.assertEqual(res[0]["call_votes"], 3)
        self.assertEqual(res[0]["put_votes"], 2)
        self.assertTrue(15 <= res[0]["suggested_exp"] <= 35)

        weak_res, weak_reason = se.resolve_confluence(cands, conflict_mode="CONC", threshold=2.5, expiry_mode="DYN")
        self.assertEqual(weak_res, [])
        self.assertIn("Слабка конфлюентність", weak_reason)

        tie = [
            {"strategy": "STARC_5M", "signal": "CALL", "suggested_exp": 15, "tf_minutes": 5, "weight": 1.0},
            {"strategy": "EMA_RIB_5M", "signal": "PUT", "suggested_exp": 15, "tf_minutes": 5, "weight": 1.0},
        ]
        tie_res, tie_reason = se.resolve_confluence(tie, conflict_mode="CONC", threshold=1.0, expiry_mode="DYN")
        self.assertEqual(tie_res, [])
        self.assertIn("Конфлікт/нічия", tie_reason)

        maj_res, _ = se.resolve_confluence(cands, conflict_mode="MAJ", threshold=1.0, expiry_mode="FIX")
        self.assertEqual(len(maj_res), 3)
        self.assertTrue(all(c["signal"] == "CALL" for c in maj_res))

    def test_new_strategies_strictly_causal_no_lookahead(self):
        rng = np.random.default_rng(123)
        n = 360
        idx = pd.date_range("2026-01-01 00:00", periods=n, freq="5min")
        close = 1.1000 + np.cumsum(rng.normal(0, 0.0007, n))
        open_ = np.roll(close, 1)
        open_[0] = 1.1000
        high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0004, n))
        low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0004, n))
        df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)

        for key in ("DIV_RSI_5M", "DIV_MACD_5M", "AO_TWIN_5M", "STARC_5M", "LINREG_5M", "ICHI_TK_5M", "PSAR_EMA_5M", "CCI_WPR_5M", "DEM_BB_5M", "WT_CROSS_5M", "JUDAS_LON_5M", "PA_3LS_5M"):
            full_s = se.evaluate_strategy_series(df, key, regime_filter=False)
            trunc_s = se.evaluate_strategy_series(df.iloc[:260], key, regime_filter=False)
            np.testing.assert_array_equal(full_s[:260], trunc_s, f"Look-ahead detected in {key}")

        idx_1m = pd.date_range("2026-01-01 00:00", periods=220, freq="1min")
        c1 = 1.1000 + np.cumsum(rng.normal(0, 0.0003, 220))
        o1 = np.roll(c1, 1)
        o1[0] = 1.1000
        df1 = pd.DataFrame({"open": o1, "high": np.maximum(o1, c1) + 0.0002, "low": np.minimum(o1, c1) - 0.0002, "close": c1, "volume": 1.0}, index=idx_1m)
        mhi_full = se.evaluate_strategy_series(df1, "PA_MHI_1M", regime_filter=False)
        mhi_trunc = se.evaluate_strategy_series(df1.iloc[:180], "PA_MHI_1M", regime_filter=False)
        np.testing.assert_array_equal(mhi_full[:180], mhi_trunc)


class AutoDisableTests(unittest.TestCase):
    def test_backtest_auto_disable_respects_threshold_sample_and_manual_override(self):
        conn = MagicMock()
        cur = conn.cursor.return_value
        cur.fetchall.return_value = [
            {"pair_name": "EUR/USD", "strategy_key": "CRSI_1M", "wins": 7, "total": 20},
            {"pair_name": "GBP/USD", "strategy_key": "CRSI_1M", "wins": 6, "total": 15},
            {"pair_name": "EUR/USD", "strategy_key": "REV_RSI14", "wins": 12, "total": 20},
            {"pair_name": "EUR/USD", "strategy_key": "STARC_5M", "wins": 2, "total": 5},
        ]
        cur.fetchone.return_value = None
        with patch.object(database, "get_connection", return_value=conn):
            events = database.evaluate_and_apply_auto_disable(source="BACKTEST", threshold=0.49, min_pair=15, min_global=30)
            disabled_pairs = {(e["pair"], e["strategy"]) for e in events["disabled_pair"]}
            self.assertIn(("EUR/USD", "CRSI_1M"), disabled_pairs)
            self.assertIn(("GBP/USD", "CRSI_1M"), disabled_pairs)
            self.assertNotIn(("EUR/USD", "REV_RSI14"), disabled_pairs)
            self.assertNotIn(("EUR/USD", "STARC_5M"), disabled_pairs)
            disabled_globals = {e["strategy"] for e in events["disabled_global"]}
            self.assertIn("CRSI_1M", disabled_globals)

        cur.fetchone.return_value = {"manual_override": True, "is_auto_disabled": False}
        with patch.object(database, "get_connection", return_value=conn):
            events2 = database.evaluate_and_apply_auto_disable(source="BACKTEST", threshold=0.49, min_pair=15, min_global=30)
            self.assertEqual(events2["disabled_pair"], [])
            self.assertEqual(events2["disabled_global"], [])


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 2, 9)
        self.db = Mock()
        self.db.has_open_reversion_signal.return_value = False
        self.db.save_signal.return_value = 1
        self.db.is_strategy_muted.return_value = (False, 0)
        self.db.get_all_bot_settings.return_value = {}
        self.telegram = Mock()
        self.telegram.send_message.return_value.message_id = 99
        self.gate = Mock()
        self.gate.check.return_value = (True, "OK")
        self.segment = Mock()
        self.segment.check.return_value = (True, "OK")
        p = patch.multiple(
            bot, bot=self.telegram, database=self.db, signal_gate=self.gate, segment_filter=self.segment,
            reversion_guard=rev.ReversionGuard(),
            strategies_status=dict.fromkeys(se.ALL_STRATEGY_KEYS, True) | {"session_filter": False, "regime_filter": True, "news_filter": True, "auto_disable_49": True},
            get_utc_now=Mock(return_value=self.now), get_live_price=Mock(return_value=1.2), schedule_signal_timer=Mock(),
        )
        p.start()
        self.addCleanup(p.stop)
        self.cand = dict(pair="EUR/USD", tf="5m", tf_minutes=5, strategy="REV_RSI14", signal="CALL", bar_end=self.now, bar_id="one", current_price=1.2, atr=0.002, rsi=12)

    def process(self, **kwargs):
        return bot.process_candidate(7, "EUR/USD", "TEST", self.cand, **kwargs)

    def test_fixed_expiry_saves_and_no_ai_ml_entry_path(self):
        self.assertTrue(self.process()[0])
        args = self.db.save_signal.call_args.kwargs
        self.assertEqual(args["expiration_mins"], 60)
        self.assertEqual(args["expires_at"], self.now + timedelta(minutes=60))
        self.assertIn("REV_RSI14", args["message_text"])
        self.gate.register.assert_called_once()
        self.assertFalse(self.process()[0])
        self.assertFalse(hasattr(ai_advisor.AIAdvisor, "evaluate_signal"))
        self.assertFalse(hasattr(bot, "ml_filter"))

    def test_news_segments_gate_adverse_freshness_and_failed_save(self):
        self.assertFalse(self.process(news_is_bad=True)[0])
        self.segment.check.return_value = (False, "segment")
        self.assertEqual(self.process(), (False, "segment"))
        self.segment.check.return_value = (True, "OK")
        self.gate.check.return_value = (False, "gate")
        self.assertEqual(self.process(), (False, "gate"))
        self.gate.check.return_value = (True, "OK")
        self.cand["bar_end"] = self.now - timedelta(seconds=181)
        self.assertFalse(self.process()[0])
        self.cand["bar_end"] = self.now
        bot.get_live_price.return_value = 1.1
        self.assertFalse(self.process()[0])
        bot.get_live_price.return_value = 1.2
        self.db.save_signal.return_value = None
        self.assertFalse(self.process()[0])
        self.gate.register.assert_not_called()

    def test_auto_manual_and_empty_error_news_paths(self):
        with patch.object(bot, "fetch_5m", return_value=pd.DataFrame()), patch.object(bot, "build_mtf_frames", return_value={"5m": pd.DataFrame()}), patch.object(bot, "generate_candidates", return_value=[self.cand]), patch.object(bot, "process_candidate", return_value=(True, "OK")) as process, patch.object(bot, "is_high_impact_news_time", return_value=(False, "OK")) as news:
            for auto in [False, True]:
                self.assertTrue(bot.analyze_single_pair(7, "EUR/USD", "TEST", auto))
            self.assertEqual(process.call_count, 2)
            bot.generate_candidates.return_value = []
            self.assertFalse(bot.analyze_single_pair(7, "EUR/USD", "TEST"))
            bot.generate_candidates.side_effect = RuntimeError("bad input")
            self.assertFalse(bot.analyze_single_pair(7, "EUR/USD", "TEST"))
            news.return_value = (True, "news")
            self.assertFalse(bot.analyze_single_pair(7, "EUR/USD", "TEST"))

    def test_settlement_edits_card_and_sends_reply_for_all_results(self):
        sig = dict(id=1, status="PENDING", ticker="TEST", entry_price=1.2, signal_type="CALL", expiration_mins=60, expires_at=self.now - timedelta(seconds=181), chat_id=7, message_id=99, message_text="original")
        self.db.get_signal_by_id.return_value = sig
        self.db.claim_signal.return_value = True
        with patch.object(bot, "get_exit_price") as price:
            for value, result in [(1.3, "WIN"), (1.1, "LOSS"), (1.2, "NEUTRAL"), (None, "VOID")]:
                self.telegram.reset_mock()
                price.return_value = value
                bot.process_signal_expiration(1)
                edited = self.telegram.edit_message_text.call_args.kwargs
                self.assertEqual(edited["message_id"], 99)
                self.assertIn("original", edited["text"])
                self.assertIn(result, edited["text"])
                self.assertEqual(self.telegram.send_message.call_args.kwargs["reply_to_message_id"], 99)
            self.telegram.edit_message_text.side_effect = RuntimeError("deleted message")
            price.return_value = 1.3
            bot.process_signal_expiration(1)
            self.assertIn("WIN", self.telegram.send_message.call_args.kwargs["text"])

    def test_daily_report_only_autoscan_chats_with_empty_and_error_cases(self):
        with patch.object(bot, "AUTO_SCAN_CHATS", {7, 8}):
            self.db.get_chat_signals.return_value = []
            bot.send_daily_report(self.now)
            self.assertEqual({c.kwargs["chat_id"] for c in self.telegram.send_message.call_args_list}, {7, 8})
            self.assertIn("немає угод", self.telegram.send_message.call_args.kwargs["text"])
            self.telegram.reset_mock()
            self.db.get_chat_signals.side_effect = RuntimeError("db offline")
            bot.send_daily_report(self.now)
            self.telegram.send_message.assert_not_called()


class CallbackTests(unittest.TestCase):
    def test_all_emitted_callbacks_and_reply_buttons_have_handlers(self):
        db = Mock()
        db.get_pair_strategy_status.return_value = True
        db.get_chat_signals.return_value = []
        db.get_auto_disabled_map.return_value = {}
        db.get_auto_disabled_items.return_value = [
            dict(scope_type="GLOBAL", pair_name="*", strategy_key="CRSI_1M", win_rate=0.40, wins=4, total=10, source="BACKTEST"),
            dict(scope_type="PAIR", pair_name="EUR/USD", strategy_key="ST_MOM_3M", win_rate=0.35, wins=7, total=20, source="LIVE"),
        ]
        db.get_detailed_pair_strategy_stats.return_value = []
        db.get_backtest_results.return_value = []
        db.get_all_bot_settings.return_value = {}
        db.clear_all_stats.return_value = True

        cfg = dict.fromkeys(se.ALL_STRATEGY_KEYS, True) | {"session_filter": False, "regime_filter": True, "news_filter": True, "auto_disable_49": True}
        with patch.multiple(bot, database=db, strategies_status=cfg, ADMIN_IDS=set(), save_strategies_config=Mock(), AUTO_SCAN_CHATS=set()), patch.object(bot.threading, "Thread"):
            markups = [
                bot.inline_main_keyboard(7),
                bot.stats_keyboard(),
                bot.clear_stats_confirm_keyboard(),
                bot.active_keyboard(),
                bot.scan_keyboard(),
                bot.check_pairs_keyboard(),
                bot.backtest_keyboard("all"),
                bot.backtest_keyboard("act"),
                bot.settings_menu_keyboard(),
                bot.thresholds_keyboard(),
                bot.confluence_keyboard(),
                bot.expiry_keyboard(),
                bot.filters_keyboard(),
                bot.tz_keyboard(),
                bot.risk_keyboard(),
                bot.strategy_keyboard(),
                bot.pairs_keyboard({}),
                bot.auto_disabled_keyboard(db.get_auto_disabled_items()),
            ]
            for cat in se.STRATEGY_CATEGORIES:
                markups.append(bot.strategy_keyboard(category=cat))
            for pair in bot.PAIRS_MAP:
                markups.append(bot.strategy_keyboard(pair=pair))
                for cat in se.STRATEGY_CATEGORIES:
                    markups.append(bot.strategy_keyboard(category=cat, pair=pair))

            callbacks = {btn.callback_data for mk in markups for row in mk.inline_keyboard for btn in row}
            for data in callbacks:
                self.assertLessEqual(len(data.encode("utf-8")), 64, data)
                u = update_for(data)
                bot.handle_callback(u, None)
                u.callback_query.answer.assert_called_once()
                u.callback_query.edit_message_text.assert_called_once()

            sig_mk = bot.signal_action_keyboard("REV_RSI14")
            for btn in [b for row in sig_mk.inline_keyboard for b in row]:
                self.assertLessEqual(len(btn.callback_data.encode("utf-8")), 64)
                u = update_for(btn.callback_data)
                bot.handle_callback(u, None)
                u.callback_query.answer.assert_called_once()

            db.get_signal_by_id.return_value = dict(id=1, chat_id=7, status="CLOSED")
            u = update_for("review|1")
            bot.handle_callback(u, None)
            bot.threading.Thread.assert_called()

            buttons = {button.text for row in bot.main_keyboard(7).keyboard for button in row} | {"🤖 Автоскан ON"}
            for text in buttons:
                u = update_for(text=text)
                bot.handle_message(u, None)
                self.assertTrue(u.effective_message.reply_text.called)
                self.assertNotEqual(u.effective_message.reply_text.call_args.args[0], "Оберіть дію з меню.")

    def test_admin_guards_global_pair_settings_and_clear_toggles(self):
        for data in [
            "toggle|REV_RSI14", "toggle|session_filter", "pair_toggle|EUR/USD|REV_RSI7",
            "pairs", "stats_clear_ask", "stats_clear_confirm", "autodis_list",
            "settings|main", "set_wr|52", "set_conf|2.0", "set_cmod|CONC", "set_emod|DYN",
            "mute1h|REV_RSI14",
        ]:
            with patch.object(bot, "ADMIN_IDS", {99}), patch.object(bot, "save_strategies_config") as save:
                u = update_for(data)
                before = dict(bot.strategies_status)
                bot.handle_callback(u, None)
                self.assertEqual(before, bot.strategies_status)
                save.assert_not_called()
                u.effective_message.reply_text.assert_called_once()

    def test_review_ownership_and_closed_only(self):
        with patch.object(bot, "database") as db, patch.object(bot.threading, "Thread") as thread:
            for sig in [None, dict(chat_id=8, status="CLOSED"), dict(chat_id=7, status="PENDING")]:
                db.get_signal_by_id.return_value = sig
                bot.handle_callback(update_for("review|1"), None)
            thread.assert_not_called()


class BacktestTests(unittest.TestCase):
    def test_optional_session_filter_comparison_and_pair_matrix_report(self):
        idx = pd.date_range("2026-01-01", periods=600, freq="5min")
        df = pd.DataFrame(dict(open=np.arange(600.0) + 1, close=np.arange(600.0) + 1, high=np.arange(600.0) + 2, low=np.arange(600.0), volume=1), index=idx)
        cfg = {p: p == "REV_RSI7" for p in rev.PRESETS}
        off = backtest.simulate_reversion_pair("EUR/USD", df, cfg)
        on = backtest.simulate_reversion_pair("EUR/USD", df, cfg | {"session_filter": True})
        self.assertTrue(off)
        self.assertTrue(on)
        self.assertLess(len(on), len(off))
        self.assertTrue(all(sessions.session_allowed(t["strategy"], t["entry_time"]) for t in on))
        with patch.object(database, "get_pair_strategy_status", return_value=True):
            compared = backtest.comparison("EUR/USD", df, cfg)
            self.assertEqual(compared[False], off)
            self.assertEqual(compared[True], on)
        self.assertNotIn("session_filter", cfg)

        report = backtest.format_pair_strategy_report(off, "Test Matrix")
        self.assertIn("Пара × Стратегія", report)
        self.assertIn("EUR/USD", report)
        self.assertIn("REV_RSI7", report)

    def test_removed_modules_have_no_importers(self):
        removed = {"indicators", "filters", "signal_pipeline", "signal_processor", "timeframe_profiles", "adaptive_optimizer", "regime_detector", "ml_model", "charts"}
        for path in Path(__file__).parent.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(node.module, removed, path.name)
                elif isinstance(node, ast.Import):
                    self.assertFalse(removed & {a.name for a in node.names}, path.name)


if __name__ == "__main__":
    unittest.main()
