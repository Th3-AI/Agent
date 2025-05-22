# Simple System Control Agent

This is a helpful agent that can control your computer based on your commands. Here are some things it can do:

- **Open websites and search online:** Just tell it the website you want to visit or what you want to search for.
- **Open applications:** You can ask it to open apps like Notepad, Chrome, or even apps it sees on your screen.
- **Get system information:** Ask about your computer's CPU usage, memory, etc.
- **Write and fix code:** It can generate code in different languages and even help fix errors in your code when you're in VSCode.
- **Write stories and content:** Ask it to write stories or other text content for you.
- **Play music/videos on YouTube:** Tell it what you want to watch or listen to on YouTube.

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

4. **Run the agent:** You can run the agent in different modes using the `agent_cli.py` script or `adk web` for the web interface.

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

## Interaction Modes

Currently, the agent primarily interacts through **text commands**. You can type your requests and the agent will respond in text.

The agent also has experimental **screen analysis** capabilities, allowing it to understand what's visible on your desktop, especially in code editors like VSCode. Potential future enhancements could include voice and video interaction, depending on the ADK setup.

**How to use:**

Just type your command or question, and the agent will try its best to understand and help you! 