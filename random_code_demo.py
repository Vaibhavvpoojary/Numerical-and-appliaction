"""
Random Code Demo
A simple demonstration of random functionality in Python
"""

import random
import datetime
import hashlib

def generate_random_numbers(count=10, min_val=1, max_val=100):
    """Generate a list of random numbers"""
    return [random.randint(min_val, max_val) for _ in range(count)]

def generate_random_string(length=10):
    """Generate a random alphanumeric string"""
    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    return ''.join(random.choice(chars) for _ in range(length))

def shuffle_list(items):
    """Shuffle a list randomly"""
    shuffled = items.copy()
    random.shuffle(shuffled)
    return shuffled

def generate_random_password(length=12):
    """Generate a random password"""
    uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    lowercase = 'abcdefghijklmnopqrstuvwxyz'
    digits = '0123456789'
    special = '!@#$%^&*'
    
    # Ensure at least one character from each category
    password = [
        random.choice(uppercase),
        random.choice(lowercase),
        random.choice(digits),
        random.choice(special)
    ]
    
    # Fill the rest with random characters
    all_chars = uppercase + lowercase + digits + special
    password.extend(random.choice(all_chars) for _ in range(length - 4))
    
    # Shuffle the password
    return ''.join(shuffle_list(password))

def generate_timestamp():
    """Generate current timestamp"""
    return datetime.datetime.now().isoformat()

def generate_hash(text):
    """Generate SHA256 hash of text"""
    return hashlib.sha256(text.encode()).hexdigest()

def main():
    print("=" * 50)
    print("Random Code Demo")
    print("=" * 50)
    
    # Generate random numbers
    print("\n1. Random Numbers:")
    numbers = generate_random_numbers(10, 1, 100)
    print(f"   {numbers}")
    
    # Generate random string
    print("\n2. Random String:")
    random_str = generate_random_string(15)
    print(f"   {random_str}")
    
    # Generate random password
    print("\n3. Random Password:")
    password = generate_random_password(16)
    print(f"   {password}")
    
    # Generate timestamp
    print("\n4. Timestamp:")
    timestamp = generate_timestamp()
    print(f"   {timestamp}")
    
    # Generate hash
    print("\n5. Hash of random string:")
    hash_value = generate_hash(random_str)
    print(f"   {hash_value}")
    
    # Shuffle example
    print("\n6. Shuffled List:")
    original = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    shuffled = shuffle_list(original)
    print(f"   Original: {original}")
    print(f"   Shuffled: {shuffled}")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    main()