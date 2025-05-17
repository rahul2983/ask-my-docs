import os
import secrets
import sys

# Function to generate a secure random secret key
def generate_secret_key(length=32):
    """Generate a cryptographically strong random key of specified length in bytes"""
    return secrets.token_hex(length)

# Function to update the .env file with the new secret key
def update_env_file(env_file='.env', secret_key=None):
    """Add or update SECRET_KEY in the .env file"""
    if secret_key is None:
        secret_key = generate_secret_key()
        
    # Check if .env file exists
    if os.path.exists(env_file):
        # Read the current contents
        with open(env_file, 'r') as f:
            lines = f.readlines()
            
        # Check if SECRET_KEY already exists
        secret_key_exists = False
        new_lines = []
        
        for line in lines:
            if line.startswith('SECRET_KEY='):
                # Replace the existing SECRET_KEY line
                new_lines.append(f'SECRET_KEY={secret_key}\n')
                secret_key_exists = True
            else:
                new_lines.append(line)
        
        # Add SECRET_KEY if it doesn't exist
        if not secret_key_exists:
            new_lines.append(f'\n# Authentication Settings\nSECRET_KEY={secret_key}\n')
        
        # Write the updated content back to the file
        with open(env_file, 'w') as f:
            f.writelines(new_lines)
            
        print(f"Updated SECRET_KEY in {env_file}")
    else:
        # Create a new .env file with the SECRET_KEY
        with open(env_file, 'w') as f:
            f.write(f'# Authentication Settings\nSECRET_KEY={secret_key}\n')
            
        print(f"Created new {env_file} file with SECRET_KEY")
    
    return secret_key

if __name__ == "__main__":
    # Get the file path from command line argument or use default
    env_file = '.env'
    if len(sys.argv) > 1:
        env_file = sys.argv[1]
    
    # Generate and update the secret key
    secret_key = generate_secret_key()
    update_env_file(env_file, secret_key)
    
    print(f"\nGenerated new SECRET_KEY: {secret_key}")
    print("Store this key securely as a backup.")