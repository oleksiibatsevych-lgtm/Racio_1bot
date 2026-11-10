import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

import ai_advisor
import backtest
import bot
import database
import reversion_engine as rev
import telegram_views as views
import trading_sessions as sessions


def update_for(data=None, text=None):
    update = Mock()
    update.effective_chat.id = 7
    update.effective_user.id = 1
    update.effective_message.text = text
    update.callback_query.data = data
    return update


class SessionTests(unittest.TestCase):
    def test_dst_session_boundaries_and_mismatched_change_weeks(self):
        cases = {
            '2026-06-01': {'06:59':'Asia','07:00':'London','11:59':'London','12:00':'Overlap','14:59':'Overlap','15:00':'NY','19:59':'NY','20:00':'Other'},
            '2026-01-02': {'06:59':'Asia','07:00':'Other','08:00':'London','12:59':'London','13:00':'Overlap','15:59':'Overlap','16:00':'NY','20:59':'NY','21:00':'Other'},
            '2026-03-10': {'07:00':'Other','08:00':'London','12:00':'Overlap','16:00':'NY','20:00':'Other'},
            '2026-10-27': {'07:00':'Other','08:00':'London','12:00':'Overlap','16:00':'NY','20:00':'Other'},
        }
        for day, times in cases.items():
            for clock, expected in times.items():
                value = datetime.fromisoformat(day + 'T' + clock)
                self.assertEqual(sessions.session_name(value), expected, (day, clock))
                self.assertEqual(sessions.session_name(value.replace(tzinfo=timezone.utc).astimezone(sessions.KYIV)), expected)

    def test_session_masks_match_vectorized_research_definition(self):
        for start in ['2026-03-01','2026-10-01']:
            idx = pd.date_range(start, periods=24*60*35, freq='min', tz='UTC')
            london=idx.tz_convert('Europe/London').hour
            ny=idx.tz_convert('America/New_York').hour
            l=(london>=8)&(london<16);n=(ny>=8)&(ny<16)
            expected=np.select([l&n,l&~n,n&~l,idx.hour<7],['Overlap','London','NY','Asia'],default='Other')
            self.assertEqual([sessions.session_name(t) for t in idx], list(expected))

    def test_selection_uses_bar_end_and_falls_through_disallowed_preset(self):
        df = pd.DataFrame(dict(open=1.,high=1.1,low=.9,close=1.,volume=1), index=pd.date_range('2026-01-01 16:20',periods=201,freq='5min'))
        ind = rev.calculate_indicators(df)
        ind.loc[ind.index[-1], list(rev.PRESETS)] = 1
        now = ind.index[-1] + timedelta(minutes=5)
        self.assertEqual(sessions.session_name(now), 'London')
        config=dict.fromkeys(rev.PRESETS,True) | {'REV_RSI14':False,'session_filter':True}
        self.assertEqual(rev.candidates_from_indicators('EUR/USD',ind,now,config)[0]['strategy'],'REV_RSI7')
        config['session_filter']=False
        self.assertEqual(rev.candidates_from_indicators('EUR/USD',ind,now,config)[0]['strategy'],'REV_BB3')

    def test_daily_report_schedule_dst_exact_time_and_period_start(self):
        for now, expected in [('2026-03-28T18:00','2026-03-28T19:00'),('2026-03-28T20:00','2026-03-29T18:00'),('2026-10-24T20:00','2026-10-25T19:00'),('2026-01-01T19:00','2026-01-02T19:00')]:
            self.assertEqual(sessions.next_daily_report(datetime.fromisoformat(now)),datetime.fromisoformat(expected).replace(tzinfo=timezone.utc))
        self.assertEqual(sessions.period_start('today',datetime(2026,6,1,22)),datetime(2026,6,1,21))
        self.assertIsNone(sessions.period_start('all',datetime(2026,1,1)))
        with self.assertRaises(ValueError):sessions.period_start('invalid',datetime(2026,1,1))


class ViewTests(unittest.TestCase):
    def test_stats_counts_bars_escape_and_empty(self):
        rows=[dict(status=status,result=result,ticker='A&B',strategy='REV_RSI14') for status,result in [('CLOSED','WIN'),('CLOSED','LOSS'),('CLOSED','NEUTRAL'),('VOID','WIN')]]
        text=views.stats_text(rows,'all')
        self.assertIn('50.0% · n=2',text)
        self.assertIn('A&amp;B',text)
        self.assertIn('█████░░░░░',text)
        self.assertIn('n=0',views.stats_text([],'today'))
        self.assertIn('Угод за цей період немає',views.stats_text([],'today'))
        for value in [-100,0,50,100,200]:self.assertEqual(len(views.progress_bar(value)),10)

    def test_card_times_stats_session_and_result_views(self):
        cand=dict(strategy='REV_RSI14',signal='CALL',rsi=12,bar_end=datetime(2026,6,1,14))
        card=views.signal_card('EUR/USD',cand,1.2,datetime(2026,6,1,14))
        for value in ['REV_RSI14','🟢 CALL','60 хв','15:00 UTC / 18:00 Київ','67.03% · n=276','Overlap']:
            self.assertIn(value,card)
        self.assertIn('VOID',views.settlement_text('VOID',None,0))
        self.assertIn('немає котирування',views.settlement_text('VOID',None,0))
        self.assertIn('Відкритих угод немає',views.active_text([],datetime(2026,1,1)))
        row=dict(id=1,ticker='EUR/USD',strategy='REV_RSI7',signal_type='PUT',expiration_mins=45,created_at=datetime(2026,1,1),expires_at=datetime(2026,1,1,0,1),status='CLOSED',result='WIN',exit_price=1.2,pips=10)
        self.assertIn('1:00',views.active_text([row],datetime(2026,1,1)))
        self.assertIn('очікуємо результат',views.active_text([row],datetime(2026,1,1,0,2)))
        self.assertIn('WIN',views.history_text([row]))
        self.assertIn('Завершених угод немає',views.history_text([]))

    def test_query_chat_scope_period_history_limit_and_connection_cleanup(self):
        connection=MagicMock();cursor=connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value=[]
        with patch.object(database,'get_connection',return_value=connection):
            database.get_chat_signals(7,'history',since=datetime(2026,1,1))
            sql,params=cursor.execute.call_args.args
            self.assertIn('chat_id = %s',sql);self.assertIn('LIMIT 15',sql)
            self.assertEqual(params,[7,datetime(2026,1,1)])
            database.get_chat_signals(7,'active')
            self.assertIn("status = 'PENDING'",cursor.execute.call_args.args[0])
            cursor.execute.side_effect=RuntimeError('DB unavailable')
            with self.assertRaises(RuntimeError):database.get_chat_signals(7)
        self.assertEqual(connection.close.call_count,3)
        with self.assertRaises(ValueError):database.get_chat_signals(7,'invalid')


class LiveTests(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,1,2,9)
        self.db=Mock();self.db.has_open_reversion_signal.return_value=False;self.db.save_signal.return_value=1
        self.telegram=Mock();self.telegram.send_message.return_value.message_id=99
        self.gate=Mock();self.gate.check.return_value=(True,'OK')
        self.segment=Mock();self.segment.check.return_value=(True,'OK')
        p=patch.multiple(bot,bot=self.telegram,database=self.db,signal_gate=self.gate,segment_filter=self.segment,
                         reversion_guard=rev.ReversionGuard(),strategies_status=dict.fromkeys(rev.PRESETS,True)|{'session_filter':False},
                         get_utc_now=Mock(return_value=self.now),get_live_price=Mock(return_value=1.2),schedule_signal_timer=Mock())
        p.start();self.addCleanup(p.stop)
        self.cand=dict(pair='EUR/USD',tf='5m',tf_minutes=5,strategy='REV_RSI14',signal='CALL',bar_end=self.now,bar_id='one',current_price=1.2,atr=.002,rsi=12)

    def process(self,**kwargs):return bot.process_candidate(7,'EUR/USD','TEST',self.cand,**kwargs)

    def test_fixed_expiry_saves_and_no_ai_ml_entry_path(self):
        self.assertTrue(self.process()[0])
        args=self.db.save_signal.call_args.kwargs
        self.assertEqual(args['expiration_mins'],60)
        self.assertEqual(args['expires_at'],self.now+timedelta(minutes=60))
        self.assertIn('REV_RSI14',args['message_text'])
        self.gate.register.assert_called_once()
        self.assertFalse(self.process()[0])
        self.assertFalse(hasattr(ai_advisor.AIAdvisor,'evaluate_signal'))
        self.assertFalse(hasattr(bot,'ml_filter'))

    def test_news_segments_gate_adverse_freshness_and_failed_save(self):
        self.assertFalse(self.process(news_is_bad=True)[0])
        self.segment.check.return_value=(False,'segment')
        self.assertEqual(self.process(),(False,'segment'))
        self.segment.check.return_value=(True,'OK');self.gate.check.return_value=(False,'gate')
        self.assertEqual(self.process(),(False,'gate'))
        self.gate.check.return_value=(True,'OK');self.cand['bar_end']=self.now-timedelta(seconds=181)
        self.assertFalse(self.process()[0])
        self.cand['bar_end']=self.now;bot.get_live_price.return_value=1.1
        self.assertFalse(self.process()[0])
        bot.get_live_price.return_value=1.2;self.db.save_signal.return_value=None
        self.assertFalse(self.process()[0]);self.gate.register.assert_not_called()

    def test_auto_manual_and_empty_error_news_paths(self):
        with patch.object(bot,'fetch_5m',return_value=pd.DataFrame()),patch.object(bot,'generate_candidates',return_value=[self.cand]),patch.object(bot,'process_candidate',return_value=(True,'OK')) as process,patch.object(bot,'is_high_impact_news_time',return_value=(False,'OK')) as news:
            for auto in [False,True]:self.assertTrue(bot.analyze_single_pair(7,'EUR/USD','TEST',auto))
            self.assertEqual(process.call_count,2)
            bot.generate_candidates.return_value=[]
            self.assertFalse(bot.analyze_single_pair(7,'EUR/USD','TEST'))
            bot.generate_candidates.side_effect=RuntimeError('bad input')
            self.assertFalse(bot.analyze_single_pair(7,'EUR/USD','TEST'))
            news.return_value=(True,'news');self.assertFalse(bot.analyze_single_pair(7,'EUR/USD','TEST'))

    def test_settlement_edits_card_and_sends_reply_for_all_results(self):
        sig=dict(id=1,status='PENDING',ticker='TEST',entry_price=1.2,signal_type='CALL',expiration_mins=60,
                 expires_at=self.now-timedelta(seconds=181),chat_id=7,message_id=99,message_text='original')
        self.db.get_signal_by_id.return_value=sig;self.db.claim_signal.return_value=True
        with patch.object(bot,'get_exit_price') as price:
            for value,result in [(1.3,'WIN'),(1.1,'LOSS'),(1.2,'NEUTRAL'),(None,'VOID')]:
                self.telegram.reset_mock();price.return_value=value
                bot.process_signal_expiration(1)
                edited=self.telegram.edit_message_text.call_args.kwargs
                self.assertEqual(edited['message_id'],99);self.assertIn('original',edited['text']);self.assertIn(result,edited['text'])
                self.assertEqual(self.telegram.send_message.call_args.kwargs['reply_to_message_id'],99)
            self.telegram.edit_message_text.side_effect=RuntimeError('deleted message')
            price.return_value=1.3;bot.process_signal_expiration(1)
            self.assertIn('WIN',self.telegram.send_message.call_args.kwargs['text'])

    def test_daily_report_only_autoscan_chats_with_empty_and_error_cases(self):
        with patch.object(bot,'AUTO_SCAN_CHATS',{7,8}):
            self.db.get_chat_signals.return_value=[]
            bot.send_daily_report(self.now)
            self.assertEqual({c.kwargs['chat_id'] for c in self.telegram.send_message.call_args_list},{7,8})
            self.assertIn('n=0',self.telegram.send_message.call_args.kwargs['text'])
            self.telegram.reset_mock();self.db.get_chat_signals.side_effect=RuntimeError('db offline')
            bot.send_daily_report(self.now);self.telegram.send_message.assert_not_called()


class CallbackTests(unittest.TestCase):
    def test_all_emitted_callbacks_and_reply_buttons_have_handlers(self):
        db=Mock();db.get_pair_strategy_status.return_value=True;db.get_chat_signals.return_value=[]
        config=dict.fromkeys(rev.PRESETS,True)|{'session_filter':False}
        with patch.multiple(bot,database=db,strategies_status=config,ADMIN_IDS=set(),save_strategies_config=Mock(),AUTO_SCAN_CHATS=set()),patch.object(bot.threading,'Thread'):
            markups=[bot.stats_keyboard(),bot.strategy_keyboard()]+[bot.strategy_keyboard(pair) for pair in bot.PAIRS_MAP]
            u=update_for('pairs');bot.handle_callback(u,None)
            markups.append(u.callback_query.edit_message_text.call_args.kwargs['reply_markup'])
            callbacks={button.callback_data for markup in markups for row in markup.inline_keyboard for button in row}
            for data in callbacks:
                u=update_for(data);bot.handle_callback(u,None)
                u.callback_query.answer.assert_called_once()
                u.callback_query.edit_message_text.assert_called_once()
            db.get_signal_by_id.return_value=dict(id=1,chat_id=7,status='CLOSED')
            u=update_for('review|1');bot.handle_callback(u,None);bot.threading.Thread.assert_called()
            buttons={button.text for row in bot.main_keyboard(7).keyboard for button in row}|{'🤖 Автоскан ON'}
            for text in buttons:
                u=update_for(text=text);bot.handle_message(u,None)
                self.assertTrue(u.effective_message.reply_text.called)
                self.assertNotEqual(u.effective_message.reply_text.call_args.args[0],'Оберіть дію з меню.')
            print(f'CALLBACK COVERAGE PASS: {len(callbacks)+1} emitted callback values; {len(buttons)} reply-button states; all dispatched.')

    def test_admin_guards_global_and_pair_toggles(self):
        for data in ['toggle|REV_RSI14','toggle|session_filter','pair_toggle|EUR/USD|REV_RSI7','pairs']:
            with patch.object(bot,'ADMIN_IDS',{99}),patch.object(bot,'save_strategies_config') as save:
                u=update_for(data);before=dict(bot.strategies_status)
                bot.handle_callback(u,None)
                self.assertEqual(before,bot.strategies_status);save.assert_not_called()
                u.effective_message.reply_text.assert_called_once()

    def test_review_ownership_and_closed_only(self):
        with patch.object(bot,'database') as db,patch.object(bot.threading,'Thread') as thread:
            for sig in [None,dict(chat_id=8,status='CLOSED'),dict(chat_id=7,status='PENDING')]:
                db.get_signal_by_id.return_value=sig
                bot.handle_callback(update_for('review|1'),None)
            thread.assert_not_called()


class BacktestTests(unittest.TestCase):
    def test_optional_session_filter_and_comparison(self):
        idx=pd.date_range('2026-01-01',periods=600,freq='5min')
        df=pd.DataFrame(dict(open=np.arange(600.)+1,close=np.arange(600.)+1,high=np.arange(600.)+2,low=np.arange(600.),volume=1),index=idx)
        config={p:p=='REV_RSI7' for p in rev.PRESETS}
        off=backtest.simulate_reversion_pair('EUR/USD',df,config)
        on=backtest.simulate_reversion_pair('EUR/USD',df,config|{'session_filter':True})
        self.assertTrue(off);self.assertTrue(on);self.assertLess(len(on),len(off))
        self.assertTrue(all(sessions.session_allowed(t['strategy'],t['entry_time']) for t in on))
        with patch.object(database,'get_pair_strategy_status',return_value=True):
            compared=backtest.comparison('EUR/USD',df,config)
            self.assertEqual(compared[False],off);self.assertEqual(compared[True],on)
        self.assertNotIn('session_filter',config)

    def test_removed_modules_have_no_importers(self):
        removed={'indicators','filters','signal_pipeline','signal_processor','timeframe_profiles','adaptive_optimizer','regime_detector','ml_model','charts'}
        for path in Path(__file__).parent.glob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node,ast.ImportFrom):self.assertNotIn(node.module,removed,path.name)
                elif isinstance(node,ast.Import):self.assertFalse(removed & {a.name for a in node.names},path.name)


if __name__ == '__main__':
    unittest.main()
