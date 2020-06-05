from __future__ import absolute_import, division, print_function, unicode_literals
import os
import hashlib
import numpy
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
# Get logging level from the environment, and default to 'DEBUG'
logger_level = os.getenv('DO_LOGGING', 'DEBUG')
logger.setLevel(logger_level)


class Encryption:

    @staticmethod
    def s():
        logging.debug('Generating secret for encryption')
        salt = '2z8876a1572e417v'
        # Generate 32-bit integer from a UUID and salt
        salted_bytestring = bytes('5d8ce0bc-fc8a-454e-9620-b736333a79e6' + salt, 'utf8')
        # Hash salted string using md5
        hash_base_16 = hashlib.md5(salted_bytestring).hexdigest()
        # Return hash value as a decimal modulo 2^32-1 as this is the max value for numpy random seed
        return str(int(hash_base_16, 16) % (2 ** 32 - 1))

    @staticmethod
    def p(np_arr):
        # Get permutation of length np_arr.shape[-1]
        permutation = Encryption.get_p(np_arr.shape[-1])
        # Apply permutation to np_arr
        return Encryption.do_p(permutation, np_arr)

    @staticmethod
    def un_p(np_arr):
        # Get permutation of length np_arr.shape[-1]
        permutation = Encryption.get_p(np_arr.shape[-1])
        # Invert permutation array
        inverse_permutation = numpy.argsort(permutation)
        logging.debug('Permutation inverted')
        # Invert permutation of np_arr
        return Encryption.do_p(inverse_permutation, np_arr)

    @staticmethod
    def get_p(length):
        logging.debug(f"Generating permutation of length {length}")
        # Get seed for permutation
        permutation_seed = int(Encryption.s())
        # Generate permutation array of length length
        numpy.random.seed(permutation_seed)
        return numpy.random.permutation(length)

    @staticmethod
    def do_p(p, np_arr):
        # Avoid modifying original array
        output_array = numpy.copy(np_arr)

        # Apply permutation array p to np_arr
        # If np_arr is 1-dimensional
        if np_arr.ndim == 1:
            logging.debug('Permuting 1 dimensional np array')
            return output_array[p]

        # If np_arr is 2(+)-dimensional, do permutation along rows
        else:
            logging.debug('Permuting each row of an np array')
            # Apply permutation to each row of np_arr
            for row_ind in range(np_arr.shape[0]):
                output_array[row_ind, ...] = np_arr[row_ind, ...][p]
            return output_array
