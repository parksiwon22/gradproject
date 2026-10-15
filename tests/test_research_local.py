import unittest
import numpy as np
import pandas as pd
from model import v5
from scripts.research_local import ROOT, metrics
from scripts.research_2024 import parse_kpx, price_rows

class ResearchTests(unittest.TestCase):
    def test_future_price_cannot_change_origin_features(self):
        raw=pd.DataFrame({'timestamp':pd.date_range('2023-01-01',periods=12*24,freq='h'),'smp':np.arange(12*24,dtype=float)})
        first=price_rows(raw)
        changed=raw.copy()
        changed.loc[changed.timestamp>='2023-01-10','smp']+=10000
        second=price_rows(changed)
        columns=[c for c in first if c not in ('actual_smp','previous_actual_smp')]
        pd.testing.assert_frame_equal(first.loc[first.target_date=='2023-01-10',columns],second.loc[second.target_date=='2023-01-10',columns])

    def test_original_v5_features_cutoff(self):
        raw=pd.read_csv(v5.DATA_PATH)
        a=v5.build_stacked_dataset(v5.preprocess_hourly(raw))
        changed=raw.copy()
        future=pd.to_datetime(changed.timestamp)>='2023-09-01'
        for c in v5.CORE_RAW_FEATURES:
            if c in changed: changed.loc[future,c]=999999
        b=v5.build_stacked_dataset(v5.preprocess_hourly(changed))
        features=v5.preprocess_hourly(raw).model_features
        np.testing.assert_allclose(a.loc[a.target_date=='2023-09-01',features],b.loc[b.target_date=='2023-09-01',features],rtol=0,atol=0)

    def test_official_parser_time_ending(self):
        block=''.join(f'c{h} = textFormmat("{h}",count);' for h in range(1,28))+'gridData.push({"Date":"2024/12/31"'
        result=parse_kpx(block)
        self.assertEqual(len(result),24)
        self.assertEqual(result.timestamp.iloc[0],pd.Timestamp('2024-12-31 00:00'))
        self.assertEqual(result.timestamp.iloc[-1],pd.Timestamp('2024-12-31 23:00'))
        self.assertEqual(result.smp.iloc[-1],24)
        with self.assertRaises(ValueError): parse_kpx('empty')
        with self.assertRaises(ValueError): parse_kpx(block+block)

    def test_oracle_regret_zero_and_false_alerts(self):
        y=np.arange(24,dtype=float)+140
        frame=pd.DataFrame({'target_date':['2023-10-01']*24,'actual_smp':y,'previous_actual_smp':np.r_[139,y[:-1]]})
        result=metrics(frame,y,150)
        self.assertEqual(result['EV_regret'],0)
        self.assertEqual(result['peak_F1'],1)
        self.assertEqual(result['false_alarms'],0)
        self.assertEqual(result['cheapest_window_overlap'],1)

if __name__=='__main__': unittest.main()
