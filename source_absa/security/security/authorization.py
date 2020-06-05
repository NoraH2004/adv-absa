from __future__ import absolute_import, division, print_function, unicode_literals
import jwt
import time
import os
from flask import request, jsonify
from functools import wraps
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
# Get logging level from the environment, and default to 'DEBUG'
logger_level = os.getenv('DO_LOGGING', 'DEBUG')
logger.setLevel(logger_level)


class Authorization:

    @staticmethod
    def public_key():
        public_key = """-----BEGIN PUBLIC KEY-----
MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEAoMNjs1nxu0YfzgGQyW90
uZVvql9EQB4iZ34uDt3A+eeJi/Ra4txZ1SYzFCiHniGE51crhcjV3lDTXdgn7Vla
DGsm0qM0oc6tZEf4T/db/GtqDAwZAqqzgItVTlt+m4QTmdMqrC4hcgh1RJ30E/QL
9JpVrOyhcFtv2HHRHWXkaJFUK7yPkdqdw8mcB11utCtrs85Ex97YpOUV1DV3pTV5
skjhFktnKkU+CTEknlhYEBGZFyOrSMbDBDuqPOz2NdPh3B6XfPx6pEz8FNftngvX
qQHQzECAQjIGhBNlpzV8RBeLfnYpWuudoOXZx7GkK2BzqbpeR9+H2WIsfvPNLpeg
noy1nWv+12I6yHvSOGyQ15BDKXlF6PQPXDU1MCqjibupXbgpSmfLp1yp+SkL1YAZ
AvXUJAE62aZ7bprPkWKhEBL9kx+fwnQYRgSvdAzuHle4T8diGyPwRtEmjmz5+oby
63KNw6MVgIjBtDouAg/+TY5/ruL4mYTbXgHdoSU53xnGGrtU/O6Thsd///n6Du8I
Dk0dC4fjfOsobI3opMmza0bQXhpkuxbSX6cTcEgo/jOe5RrWdyppxP3KNLw/BxOZ
g1ivogRvvXiGRxddwGPJ1Rj4iLYkIdRODZgp9WpJNAi9gNCuYmB6qeGCPF97bKVI
ciqFIbxYKvCSFP18KYBq+20CAwEAAQ==
-----END PUBLIC KEY-----"""
        return public_key

    @staticmethod
    def authorize(token, functionality=None, language=None):
        logger.debug(f"Running authorization for token for functionality {functionality} and language {language}")
        try:
            payload = jwt.decode(token, Authorization.public_key(), algorithms=['RS256'])
            # Verify functionality permissions
            if functionality is not None:
                # get the substring after the initial '/'
                functionality_domain = functionality.split('/', 1)[0]
                if 'functionality' not in payload:
                    logger.warning(
                        'Functionality argument not in token. The token is probably outdated and should be regenerated.')
                    return {'verified': False, 'message': 'Functionality not in token', 'info': payload}
                if functionality not in payload['functionality'] \
                        and (functionality_domain + '/*') not in payload['functionality'] \
                        and '*/*' not in payload['functionality']:
                    return {'verified': False, 'message': 'Unauthorized functionality', 'info': payload}
            # Verify language permissions
            if language is not None:
                if 'languages' not in payload:
                    logger.warning(
                        'Languages argument not in token. The token is probably outdated and should be regenerated.')
                    return {'verified': False, 'message': 'Languages not in token', 'info': payload}
                elif language not in payload['languages'] and '*' not in payload['languages']:
                    return {'verified': False, 'message': 'Unauthorized language', 'info': payload}
            # Otherwise, the token is valid
            return {'verified': True, 'message': 'OK', 'info': payload}
        except jwt.ExpiredSignatureError:
            logger.warning('Token has expired')
            return {'verified': False, 'message': 'Signature expired', 'info': None}
        except jwt.InvalidSignatureError:
            logger.warning('Invalid token signature')
            return {'verified': False, 'message': 'Invalid signature', 'info': None}
        except jwt.DecodeError:
            logger.warning('Token could not be decoded')
            return {'verified': False, 'message': 'Token could not be decoded', 'info': None}

    @staticmethod
    def generate_token(username, hours, key, languages=[], functionality=[]):
        exp = int(time.time() + hours * 3600)
        private_key = key
        logger.debug(f"Generating token for languages {languages} with functionality {functionality}")
        return jwt.encode({'exp': exp,
                           'sub': username,
                           'iss': 'deepopinion.ai',
                           'languages': languages,
                           'functionality': functionality,
                           'iat': int(time.time())}, private_key, algorithm='RS256'
                          ).decode('utf-8')


def authorize_flask_request(functionality):
    """Decorator function to authorize Flask API calls

        See here for information on decorator functions with arguments: http://scottlobdell.me/2015/04/decorators-arguments-python/
        The inner function 'authorization_decorator' is a normal decorator function,
        with the authorize_flask_request wrapper used to pass in the functionality argument

        Use in a Flask API like:
        @app.route("/route", methods=['POST', 'GET'])
        @authorize_flask_request('absa-api/predict')
        def predict():
        ...
    """

    logger.debug(f"Running flask request authorization for functionality {functionality}")

    def authorization_decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if 'Authorization' not in request.headers:
                return jsonify({'message': 'Unauthorized'}), 401
            data = request.headers['Authorization']
            token = str.replace(str(data), 'Bearer ', '')
            authorization_response = Authorization.authorize(token, functionality=functionality)
            if not authorization_response['verified']:
                return jsonify({'message': authorization_response['message']}), 403
            return f(*args, **kwargs)

        return wrapper

    return authorization_decorator
