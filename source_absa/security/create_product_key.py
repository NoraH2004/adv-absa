from security import Authorization
import os

days = input("How many days should the key be valid? ")
name = input("Name of consumer that uses the key? ")


functionality = input("Add comma separated functionalities, the key should provide:  ")
functionality = [f.strip() for f in functionality.split(",")]
product_key = Authorization.generate_token(name, int(days)*24.0,  os.environ['DO_KEY'], languages=['*'], functionality=functionality)
print()
print("#####################")
print()
print(product_key)
print()
print("#####################")
print()
print("Please the key here: https://jwt.io/ to check the everything is right.")
print()
