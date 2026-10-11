"""Re-run three valid and one invalid interface examples using Streamlit AppTest."""
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest
base=Path(__file__).resolve().parent
results=[]
for example in ['Example 1','Example 2','Example 3']:
    at=AppTest.from_file(str(base/'app.py'),default_timeout=30).run()
    assert not at.exception
    next(x for x in at.selectbox if x.label=='Starting profile').select(example).run()
    next(x for x in at.button if x.label=='Estimate access need').click().run()
    assert not at.exception
    results.append({'example':example,'success':[x.value for x in at.success],'metric':[x.value for x in at.metric]})
    assert at.success
at=AppTest.from_file(str(base/'app.py'),default_timeout=30).run()
next(x for x in at.number_input if x.label=='Improved water access in 2011 (%)').set_value(150)
next(x for x in at.button if x.label=='Estimate access need').click().run()
assert not at.exception
assert any('between 0 and 100' in x.value for x in at.error)
results.append({'invalid_case':[x.value for x in at.error]})
print(json.dumps(results,indent=2))
