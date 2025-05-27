import os
import sys
import getpass
from pathlib import Path
import re

def validate_email(email):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def get_email_provider(email):
    """Get email provider from email address."""
    domain = email.split('@')[1].lower()
    if 'gmail' in domain:
        return 'gmail'
    elif 'outlook' in domain or 'hotmail' in domain or 'office365' in domain:
        return 'outlook'
    elif 'yahoo' in domain:
        return 'yahoo'
    return 'other'

def get_smtp_settings(provider):
    """Get SMTP settings based on email provider."""
    settings = {
        'gmail': {
            'server': 'smtp.gmail.com',
            'port': '587',
            'instructions': """
For Gmail, you need to:
1. Enable 2-Step Verification in your Google Account
2. Generate an App Password:
   - Go to your Google Account settings
   - Security > App Passwords
   - Select 'Mail' and your device
   - Use the generated 16-character password
"""
        },
        'outlook': {
            'server': 'smtp.office365.com',
            'port': '587',
            'instructions': """
For Outlook:
1. Use your regular email password
2. Make sure your account allows SMTP access
3. If using 2FA, you may need to generate an app password
"""
        },
        'yahoo': {
            'server': 'smtp.mail.yahoo.com',
            'port': '587',
            'instructions': """
For Yahoo:
1. Enable "Allow apps that use less secure sign in"
2. You may need to generate an app password if using 2FA
"""
        },
        'other': {
            'server': '',
            'port': '587',
            'instructions': """
For other email providers:
1. Check your email provider's SMTP settings
2. You may need to enable SMTP access in your account settings
3. Use the appropriate SMTP server and port
"""
        }
    }
    return settings.get(provider, settings['other'])

def create_env_file():
    """Create or update .env file with email settings."""
    print("Email Configuration Setup")
    print("========================")
    
    # Get email address
    while True:
        email = input("\nEnter your email address: ").strip()
        if validate_email(email):
            break
        print("Invalid email format. Please try again.")
    
    # Get email provider and settings
    provider = get_email_provider(email)
    settings = get_smtp_settings(provider)
    
    # Show provider-specific instructions
    print(settings['instructions'])
    
    # Get password
    password = getpass.getpass("\nEnter your email password or app password: ")
    
    # Create .env content
    env_content = f"""# Email Configuration
# ==================

# SMTP Server Settings
SMTP_SERVER={settings['server']}
SMTP_PORT={settings['port']}

# Sender Email Settings
SENDER_EMAIL={email}

# Sender Password Settings
SENDER_PASSWORD={password}

# Additional Notes:
# ----------------
# 1. Keep this file secure and never commit it to version control
# 2. For Gmail users: Make sure you're using an App Password
# 3. For Outlook users: Ensure SMTP access is enabled
# 4. For Yahoo users: Enable "Allow apps that use less secure sign in"
"""
    
    # Write to .env file
    env_path = Path('.env')
    if env_path.exists():
        backup_path = Path('.env.backup')
        if not backup_path.exists():
            env_path.rename(backup_path)
            print("\nExisting .env file backed up to .env.backup")
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print("\nEmail configuration has been saved to .env file")
    print("You can now use the email functionality in the agent.")

def main():
    try:
        create_env_file()
    except KeyboardInterrupt:
        print("\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 