import requests
import json
import os
import pytest

# Prepare
os.system('. env/bin/activate')
os.system('git pull')
os.system('pip install -e . --extra-index-url=https://${PIP_PULL}:@pypi.fury.io/deepopinion')

# Run tests
result = pytest.main(['absa/tests', '-s'])

if result == pytest.ExitCode.TESTS_FAILED:
    # Notify slack
    headers = {'Content-type': 'application/json'}
    data = {
        "text": "*ABSA Module Test Runner!*",
        "attachments":
        [
            {
                "color": "#D10C20",
                "text": "ABSA tests are failing! Action is required!"
            }
        ]
    }
    response = requests.post(url='https://hooks.slack.com/services/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',
                             headers=headers,
                             data=json.dumps(data))
else:
    # Notify slack
    headers = {'Content-type': 'application/json'}
    data = {
        "text": "*ABSA Module Test Runner!*",
        "attachments":
        [
            {
                "color": "#41AA58",
                "text": "ABSA tests are running successfully! Great job!"
            }
        ]
    }
    response = requests.post(url='https://hooks.slack.com/services/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',
                             headers=headers,
                             data=json.dumps(data))
