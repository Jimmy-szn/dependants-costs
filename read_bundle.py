import os, joblib, json, sys
path = os.path.abspath('c:/Users/jimmy/dependants-costs/models/bae_bundle.joblib')
print('Bundle path:', path)
if not os.path.exists(path):
    print('Bundle not found')
    sys.exit(1)
bundle = joblib.load(path)
print('Keys:', list(bundle.keys()))
print('Stats keys:', list(bundle.get('stats', {}).keys()))
print('Stats content:', json.dumps(bundle.get('stats', {}), indent=2))
