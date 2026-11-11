import copy
import json
from pathlib import Path
import pytest
from volrisk.copilot import RiskEvidence,explain,PROTOCOLS

@pytest.fixture(scope='module')
def evidence():
    root=Path(__file__).resolve().parents[1]
    artifacts=[root/'results/ROBUSTNESS_V2.csv',*[root/f'results/robustness/v2/{p}/predictions.csv' for p in PROTOCOLS]]
    if not all(p.exists() for p in artifacts): pytest.skip('Generate local V2 empirical forecasts before running artifact integration tests')
    return RiskEvidence(root)

def test_future_outcomes_absent_until_revealed(evidence):
    context=evidence.context('static_matched','2024-01-02','LSTM_market','AAPL')
    raw=json.dumps(context)
    assert 'selected_actual_volatility_pct' not in raw
    assert 'actual_top5' not in raw
    assert 'model_top5_overlap' not in raw
    assert 'qlike' not in raw.lower()  # Later-period scores cannot guide ex-ante review.
    revealed=evidence.context('static_matched','2024-01-02','LSTM_market','AAPL',True)
    assert revealed['evidence'][-1]['id']=='E7'

def test_watchlist_uses_selected_model_not_future_labels(evidence):
    context=evidence.context('static_matched','2024-01-02','LSTM_market','AAPL')
    observed=context['evidence'][1]['data']
    actual=evidence.frames['static_matched'].query('date == "2024-01-02"').sort_values(['LSTM_market','ticker'],ascending=[False,True]).head(5)
    assert [r['ticker'] for r in observed]==actual.ticker.tolist()
    assert len(observed)==5

def test_holdout_context_uses_its_own_cohort(evidence):
    c=evidence.context('unseen_companies','2024-01-02','LSTM_market','AAPL')
    assert c['evidence'][0]['data']['cohort_size']==8
    assert evidence.day_payload('unseen_companies','2024-01-02','LSTM_market')['random_top5_overlap']==.625

def test_invalid_selection_is_rejected(evidence):
    with pytest.raises(ValueError): evidence.context('static_matched','2099-01-01','LSTM_market','AAPL')
    with pytest.raises(ValueError): evidence.context('unseen_companies','2024-01-02','LSTM_market','NVDA')

def response(brief,status=200):
    class Reply:
        status_code=status
        def json(self): return {'id':'mock-only','status':'completed','output':[{'content':[{'type':'output_text','text':json.dumps(brief)}]}]}
    return Reply()

BRIEF={'summary':'Review the selected evidence.','findings':[{'text':'Models disagree on relative risk.','evidence_ids':['E4']}],'next_steps':['Inspect public filings before making a decision.'],'limitations':['Price direction is unavailable.']}

def test_real_api_contract_with_mock_transport_and_no_retention(evidence):
    context=evidence.context('static_matched','2024-01-02','LSTM_market','AAPL')
    def transport(url,**kwargs):
        assert url=='https://api.openai.com/v1/responses'
        assert kwargs['json']['store'] is False
        assert kwargs['json']['text']['format']['strict'] is True
        assert 'actual_top5' not in kwargs['json']['input']
        return response(BRIEF)
    result=explain('Explain this view',context,'test-placeholder',transport=transport)
    assert result['brief']==BRIEF
    assert 'test-placeholder' not in json.dumps(result)

@pytest.mark.parametrize('bad', ['bad_citation','numeric_prose'])
def test_unsupported_citations_and_numbers_are_blocked(evidence,bad):
    brief=copy.deepcopy(BRIEF)
    if bad=='bad_citation': brief['findings'][0]['evidence_ids']=['E7']
    else: brief['summary']='Volatility will rise 99%.'
    with pytest.raises(RuntimeError):
        explain('Explain',evidence.context('static_matched','2024-01-02','LSTM_market','AAPL'),'test-placeholder',transport=lambda *a,**k:response(brief))

def test_missing_key_never_creates_fake_answer(evidence):
    with pytest.raises(RuntimeError,match='not configured'):
        explain('Explain',evidence.context('static_matched','2024-01-02','LSTM_market','AAPL'),'')

def test_provider_error_does_not_echo_secrets(evidence):
    with pytest.raises(RuntimeError,match='rejected this API key') as error:
        explain('Explain',evidence.context('static_matched','2024-01-02','LSTM_market','AAPL'),'secret-placeholder',transport=lambda *a,**k:response({},401))
    assert 'secret-placeholder' not in str(error.value)
