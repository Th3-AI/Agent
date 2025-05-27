# Simple System Control Agent

This is a helpful agent that can control your computer based on your commands. Here are some things it can do:

- **Open websites and search online:** Just tell it the website you want to visit or what you want to search for.
- **Open applications:** You can ask it to open apps like Notepad, Chrome, or other applications on your system.
- **Get system information:** Ask about your computer's CPU usage, memory, etc.
- **Screen analysis:** The agent can analyze what's on your screen, recognize elements, and interact with applications based on visual content.
- **Write and fix code:** It can generate code in different languages and help fix errors in your code when you're in VSCode.
- **Write stories and content:** Ask it to write stories or other text content for you.
- **Play music/videos on YouTube:** Tell it what you want to watch or listen to on YouTube.
- **Schedule tasks:** Set up reminders or schedule the agent to perform tasks at specific times.
- **Send emails:** Compose and send professional emails with proper formatting and attachments.
- **Multitasking capabilities:** The agent can handle multiple tasks simultaneously.
- **Remember context and preferences:** With contextual memory, the agent remembers your past interactions and builds a memory of facts about you.
- **System control:** Adjust volume and screen brightness, get video information, and create YouTube playlists.

## Setup

To get started with the agent, follow these steps:

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Install dependencies:** Make sure you have Python installed. Then install the required packages using the `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API Key:** You'll need a Google Generative AI API key. Create a `.env` file in the project root and add your API key like this:
   ```
   GOOGLE_API_KEY=YOUR_API_KEY_HERE
   ```
   Replace `YOUR_API_KEY_HERE` with your actual API key.

4. **Configure Email Settings:** Add your email configuration to the `.env` file:
   ```
   SMTP_SERVER=smtp.gmail.com
   SMTP_PORT=587
   SENDER_EMAIL=your.email@gmail.com
   SENDER_PASSWORD=your_app_password_here
   ```
   Note: For Gmail users, you'll need to use an App Password. Enable 2-Step Verification in your Google Account and generate an App Password for "Mail".

5. **Run the agent:** You can run the agent in different modes using the `agent_cli.py` script or `adk web` for the web interface.

   - **Text Mode:** Interact with the agent using text commands in your terminal.
     ```bash
     python agent_cli.py text
     ```

   - **Voice Mode:** Interact with the agent using your voice (requires microphone setup).
     ```bash
     python agent_cli.py voice
     ```

   - **Web Mode:** Access the agent through a web-based user interface (requires ADK setup).
     ```bash
     adk web
     ```
     *(Note: Make sure you are in the correct directory and have followed the ADK setup instructions if using `adk web`.)*

## Example Use Cases

### System Control Commands
```bash
# Volume control
set volume to 75
what's the current volume

# Screen brightness
set brightness to 80
what's the current brightness
```

### YouTube Features
```bash
# Get video information
get info for https://www.youtube.com/watch?v=VIDEO_ID

# Create playlist
create playlist named "My Favorites" with https://www.youtube.com/watch?v=VIDEO1 https://www.youtube.com/watch?v=VIDEO2

# Search and play videos
search pokemon on youtube
youtube cooking recipes
```

### Email Commands
The agent can help you compose and send professional emails with proper formatting:

```bash
# Send a simple email
email john@example.com that I can't come tomorrow due to bad health

# Send an email with a subject
email john@example.com about "Meeting Cancellation" that I need to reschedule our meeting

# Send an email with attachments
email john@example.com about "Project Report" that Here's the latest report with file "report.pdf"

# Send a formatted email with multiple attachments
email team@company.com about "Weekly Update" that Please find attached the weekly reports with files "report1.pdf" "report2.xlsx"
```

### Screen Analysis
```bash
# Analyze current screen
analyze screen

# Find specific elements
find button on screen
find text "Submit" on screen
```

### Task Scheduling
```bash
# Schedule a task
schedule 'check email' at '3pm tomorrow'

# Schedule a recurring task
schedule 'backup files' at '2am' repeat 'daily'

# List scheduled tasks
list tasks

# Remove a task
remove task 1
```

### Code Generation
```bash
# Generate Python code
write python code for a simple calculator

# Generate HTML page
write html page for a portfolio website
```

### System Control
```bash
# Open applications
open notepad
open chrome

# Search web
search for python tutorials
search youtube for cooking recipes

# Get system info
system info
show cpu usage

# Control volume and brightness
set volume to 50
set brightness to 75
```

## System Control Features

The agent provides advanced system control capabilities:

- **Volume Control:** Adjust system volume levels (0-100%)
- **Screen Brightness:** Control screen brightness (0-100%)
- **YouTube Integration:** 
  - Get detailed video information (title, views, upload date)
  - Create and manage playlists
  - Search and play videos
  - Get video statistics

**System control commands:**
- `set volume to [0-100]` - Adjust system volume
- `set brightness to [0-100]` - Adjust screen brightness
- `get info for [youtube_url]` - Get video information
- `create playlist named [name] with [video_urls]` - Create YouTube playlist

## Email Features

The agent provides advanced email capabilities:

- **Professional Formatting:** Automatically formats emails with proper greetings and sign-offs
- **Smart Subject Generation:** Generates appropriate subject lines based on email content
- **Attachment Support:** Can attach multiple files of various types (PDF, DOC, images, etc.)
- **Context-Aware:** Generates appropriate email content based on the context
- **Multiple Email Providers:** Supports Gmail, Outlook, Yahoo, and other SMTP servers

**Email commands:**
- `email [address] that [message]` - Send a simple email
- `email [address] about [subject] that [message]` - Send an email with subject
- `email [address] about [subject] that [message] with file [filename]` - Send email with attachment

## Screen Analysis Capabilities

The agent has powerful screen analysis features that allow it to:

- **Understand screen content:** Using computer vision and OCR technologies, the agent can "see" and understand what's on your screen.
- **Recognize UI elements:** Identify buttons, text fields, icons, and other UI components.
- **Read text from screen:** Extract and process text from any visible application using Tesseract OCR.
- **Analyze code editors:** Particularly useful in environments like VSCode, where it can understand code context.
- **Identify applications:** Recognize which applications are open and visible.
- **Assist with visual tasks:** Help you find elements on screen or guide you through complex interfaces.

**Screen analysis commands:**
- `analyze screen` - Analyze what's currently visible on your screen
- `find [element] on screen` - Look for a specific UI element or text
- `describe what you see` - Get a description of the current screen content

## MultiTasking Capabilities

The agent supports multitasking, allowing you to:

- **Run multiple tasks simultaneously:** The agent can handle several operations at once.
- **Schedule tasks for later:** Set reminders or schedule actions to occur at specific times.
- **Manage scheduled tasks:** View, modify, or cancel previously scheduled tasks.

**Task scheduling commands:**
- `schedule [task] at [time]` - Schedule a task to run at a specific time
- `show scheduled tasks` - View all currently scheduled tasks
- `cancel task [task name or number]` - Cancel a scheduled task

## Contextual Memory

The agent is equipped with a contextual memory system that allows it to:

- **Remember your conversations:** The agent stores recent conversations to provide more context-aware responses.
- **Learn about you:** The agent automatically extracts facts about you from your conversations (like your name, interests, and preferences).
- **Personalize responses:** Over time, the agent tailors its responses based on what it knows about you.
- **Maintain topic awareness:** The agent can remember information about topics you've discussed before.

**Memory commands:**
- `show memory` or `show what you remember` - See what the agent has stored in its memory
- `clear memory` - Reset the agent's memory if needed

The contextual memory is saved between sessions, so the agent will remember you even after restarting.

## Using the Agent

Just type your command or question, and the agent will try its best to understand and help you! The more you interact with it, the better it will understand your preferences and needs.

Common commands:
- `open [website or application]` - Opens a website or installed application
- `search for [query]` - Searches the web for your query
- `write [content type] about [topic]` - Generates written content
- `system info` - Shows information about your computer system
- `email [address] that [message]` - Sends an email
- `schedule [task] at [time]` - Schedules a task
- `analyze screen` - Analyzes current screen content 