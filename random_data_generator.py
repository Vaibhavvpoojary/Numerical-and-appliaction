"""
Random Data Generator
Generates various types of random data for testing purposes
"""

import random
import string
import json
from datetime import datetime, timedelta
from faker import Faker

# Initialize Faker for generating fake data
fake = Faker()

def generate_random_email():
    """Generate a random email address"""
    return fake.email()

def generate_random_name():
    """Generate a random full name"""
    return fake.name()

def generate_random_address():
    """Generate a random address"""
    return {
        'street': fake.street_address(),
        'city': fake.city(),
        'state': fake.state(),
        'zipcode': fake.zipcode(),
        'country': fake.country()
    }

def generate_random_phone():
    """Generate a random phone number"""
    return fake.phone_number()

def generate_random_date(start_date=None, end_date=None):
    """Generate a random date between start and end"""
    if not start_date:
        start_date = datetime.now() - timedelta(days=365)
    if not end_date:
        end_date = datetime.now()
    
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    random_date = start_date + timedelta(days=random_days)
    return random_date.strftime('%Y-%m-%d')

def generate_random_ip():
    """Generate a random IP address"""
    return fake.ipv4()

def generate_random_user_agent():
    """Generate a random user agent string"""
    return fake.user_agent()

def generate_random_credit_card():
    """Generate random credit card details"""
    return {
        'number': fake.credit_card_number(),
        'expiry': fake.credit_card_expire(),
        'provider': fake.credit_card_provider()
    }

def generate_random_company():
    """Generate random company information"""
    return {
        'name': fake.company(),
        'catch_phrase': fake.catch_phrase(),
        'bs': fake.bs()
    }

def generate_random_job():
    """Generate random job information"""
    return {
        'title': fake.job(),
        'company': fake.company()
    }

def generate_random_color():
    """Generate a random hex color code"""
    return fake.hex_color()

def generate_random_uuid():
    """Generate a random UUID"""
    return fake.uuid4()

def generate_random_latitude_longitude():
    """Generate random latitude and longitude"""
    return {
        'latitude': fake.latitude(),
        'longitude': fake.longitude()
    }

def generate_sample_user():
    """Generate a complete random user profile"""
    return {
        'id': random.randint(1000, 9999),
        'name': generate_random_name(),
        'email': generate_random_email(),
        'phone': generate_random_phone(),
        'address': generate_random_address(),
        'date_of_birth': generate_random_date(
            datetime(1950, 1, 1),
            datetime(2005, 12, 31)
        ),
        'joined': generate_random_date(
            datetime(2020, 1, 1),
            datetime.now()
        ),
        'ip_address': generate_random_ip()
    }

def generate_sample_transaction():
    """Generate a random transaction"""
    return {
        'transaction_id': fake.uuid4(),
        'user_id': random.randint(1000, 9999),
        'amount': round(random.uniform(10.0, 1000.0), 2),
        'currency': random.choice(['USD', 'EUR', 'GBP', 'INR']),
        'timestamp': datetime.now().isoformat(),
        'status': random.choice(['pending', 'completed', 'failed']),
        'payment_method': random.choice(['credit_card', 'debit_card', 'paypal', 'bank_transfer'])
    }

def main():
    print("=" * 60)
    print("Random Data Generator")
    print("=" * 60)
    
    print("\n1. Random User Profile:")
    user = generate_sample_user()
    print(json.dumps(user, indent=2))
    
    print("\n2. Random Transaction:")
    transaction = generate_sample_transaction()
    print(json.dumps(transaction, indent=2))
    
    print("\n3. Random Company:")
    company = generate_random_company()
    print(json.dumps(company, indent=2))
    
    print("\n4. Random Job:")
    job = generate_random_job()
    print(json.dumps(job, indent=2))
    
    print("\n5. Random Credit Card (test data):")
    card = generate_random_credit_card()
    print(json.dumps(card, indent=2))
    
    print("\n6. Random Location:")
    location = generate_random_latitude_longitude()
    print(json.dumps(location, indent=2))
    
    print("\n7. Random Color:")
    color = generate_random_color()
    print(f"   {color}")
    
    print("\n8. Random UUID:")
    uuid = generate_random_uuid()
    print(f"   {uuid}")
    
    print("\n" + "=" * 60)
    print("Data generation completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()