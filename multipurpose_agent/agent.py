import os
import asyncio
import webbrowser
import subprocess
import platform
import psutil
import tempfile
import time
import numpy as np
import cv2
import pyautogui
from PIL import Image
import pytesseract
from pathlib import Path
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import json

# Configure API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyB0XSUK7f3LhjG9i2n8frZSB4nrKAgLEg0")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Configure genai
genai.configure(api_key=GOOGLE_API_KEY)

def execute_system_command(command: str) -> str:
    """Execute a system command and return the output."""
    try:
        # Use shell=True for Windows commands
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        return ""
    except Exception as e:
        print(f"Error executing command: {e}")
        return ""

def open_browser(url: str = "https://www.google.com") -> str:
    """Open a URL in the default browser."""
    try:
        # Ensure URL has proper format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        webbrowser.open(url)
        return f"I've opened {url} for you"
    except Exception as e:
        print(f"Error opening browser: {e}")
        return "I've opened your default browser"

def get_system_info() -> str:
    """Get basic system information."""
    try:
        info = {
            "OS": platform.system(),
            "OS Version": platform.version(),
            "CPU Usage": f"{psutil.cpu_percent()}%",
            "Memory Usage": f"{psutil.virtual_memory().percent}%",
            "Disk Usage": f"{psutil.disk_usage('/').percent}%"
        }
        return "\n".join(f"{k}: {v}" for k, v in info.items())
    except Exception as e:
        print(f"Error getting system info: {e}")
        return "Here's your system information"

def create_and_write_file(content: str, extension: str = ".txt") -> str:
    """Create a temporary file with the given content and return its path."""
    try:
        # Create a more descriptive filename
        prefix = f"generated_{int(time.time())}"
        with tempfile.NamedTemporaryFile(suffix=extension, prefix=prefix, delete=False, mode='w', encoding='utf-8') as f:
            f.write(content)
            return f.name
    except Exception as e:
        print(f"Error creating file: {e}")
        return ""

def open_in_editor(file_path: str) -> str:
    """Open a file in the default editor."""
    try:
        if platform.system() == "Windows":
            # Try VSCode first
            vscode_paths = [
                "C:\\Program Files\\Microsoft VS Code\\Code.exe",
                "C:\\Program Files (x86)\\Microsoft VS Code\\Code.exe",
                os.path.expanduser("~\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe")
            ]
            
            for vscode_path in vscode_paths:
                if os.path.exists(vscode_path):
                    subprocess.Popen([vscode_path, file_path])
                    return f"I've opened the file in VSCode"
            
            # Fallback to notepad
            subprocess.Popen(["notepad", file_path])
            return f"I've opened the file in Notepad"
        elif platform.system() == "Darwin":  # macOS
            # Try VSCode first
            subprocess.Popen(["open", "-a", "Visual Studio Code", file_path])
            return f"I've opened the file in VSCode"
        else:  # Linux
            # Try VSCode first
            subprocess.Popen(["code", file_path])
            return f"I've opened the file in VSCode"
    except Exception as e:
        print(f"Error opening editor: {e}")
        # Fallback to default editor
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["notepad", file_path])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", "-e", file_path])
            else:
                subprocess.Popen(["xdg-open", file_path])
            return "I've opened the file for you"
        except Exception as e:
            print(f"Error in fallback editor: {e}")
            return "I've opened the file for you"

def generate_code(task: str, language: str) -> str:
    """Generate code based on the task description."""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        if language.lower() == "html":
            prompt = f"""Generate a complete HTML file with CSS and JavaScript for: {task}. 
            Include all necessary HTML, CSS, and JavaScript in a single file.
            Use modern HTML5, CSS3, and JavaScript features.
            Make it visually appealing and interactive.
            Only provide the code, no explanations, no code blocks, no backticks."""
        else:
            prompt = f"Generate a simple {language} code for: {task}. Only provide the code, no explanations, no code blocks, no backticks."
        
        response = model.generate_content(prompt)
        # Clean up the response to remove any markdown formatting
        code = response.text.strip()
        if code.startswith('```'):
            code = code.split('\n', 1)[1]
        if code.endswith('```'):
            code = code.rsplit('\n', 1)[0]
        return code.strip()
    except Exception as e:
        print(f"Error generating code: {e}")
        return ""

def run_python_file(file_path: str) -> str:
    """Run a Python file and return its output."""
    try:
        result = subprocess.run(['python', file_path], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running Python file: {e}")
        return f"Error: {e.stderr}"
    except Exception as e:
        print(f"Error running Python file: {e}")
        return "Error running the Python file"

def open_html_in_browser(file_path: str) -> str:
    """Open an HTML file in the default browser."""
    try:
        # Convert to absolute path
        abs_path = os.path.abspath(file_path)
        # Convert to file URL
        file_url = f"file:///{abs_path.replace(os.sep, '/')}"
        webbrowser.open(file_url)
        return f"I've opened the HTML file in your browser"
    except Exception as e:
        print(f"Error opening HTML file: {e}")
        return "I've opened the file in your browser"

def setup_chrome_driver():
    """Set up and return a Chrome WebDriver instance."""
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Try to find Chrome in common locations
        chrome_paths = [
            "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
            "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
            os.path.expanduser("~\\AppData\\Local\\Google\\Chrome\\Application\\chrome.exe")
        ]
        
        chrome_path = None
        for path in chrome_paths:
            if os.path.exists(path):
                chrome_path = path
                break
        
        if chrome_path:
            chrome_options.binary_location = chrome_path
        
        service = Service()
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver
    except Exception as e:
        print(f"Error setting up Chrome driver: {e}")
        return None

def play_youtube_video(search_query: str) -> str:
    """Search YouTube and play the first video result."""
    try:
        driver = setup_chrome_driver()
        if not driver:
            # Fallback to just opening the search results
            url = f"https://www.youtube.com/results?search_query={search_query}"
            webbrowser.open(url)
            return f"I've opened YouTube and searched for {search_query}"
        
        # Search for the video
        driver.get(f"https://www.youtube.com/results?search_query={search_query}")
        
        # Wait for the first video to be clickable
        wait = WebDriverWait(driver, 10)
        first_video = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "ytd-video-renderer a#video-title"))
        )
        
        # Get the video URL and title
        video_url = first_video.get_attribute("href")
        video_title = first_video.get_attribute("title")
        
        # Close the headless browser
        driver.quit()
        
        # Open the video in the default browser
        webbrowser.open(video_url)
        return f"I've found and started playing: {video_title}"
        
    except Exception as e:
        print(f"Error playing YouTube video: {e}")
        # Fallback to just opening the search results
        url = f"https://www.youtube.com/results?search_query={search_query}"
        webbrowser.open(url)
        return f"I've opened YouTube and searched for {search_query}"

def capture_screen():
    """Capture the current screen and return the image."""
    try:
        screenshot = pyautogui.screenshot()
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error capturing screen: {e}")
        return None

def analyze_screen():
    """Analyze the current screen and return information about visible elements."""
    try:
        # Capture screen
        screen = capture_screen()
        if screen is None:
            return "Unable to capture screen"
        
        # Save screen for analysis
        temp_path = os.path.join(tempfile.gettempdir(), "screen_analysis.png")
        cv2.imwrite(temp_path, screen)
        
        # Use AI to analyze the screen
        model = genai.GenerativeModel('gemini-1.5-flash')
        image_parts = [
            {
                "mime_type": "image/png",
                "data": open(temp_path, "rb").read()
            }
        ]
        
        prompt = """Analyze this screenshot in detail and provide a comprehensive analysis. Focus on:
        1. What application or window is currently visible (be specific)
        2. Any visible text, code, or content
        3. Any visible icons, buttons, or interactive elements
        4. The current context and what the user might be doing
        5. Any visible URLs or website content
        6. Any visible file names or paths
        
        Respond in this exact JSON format:
        {
            "current_app": "detailed name of visible application",
            "visible_text": "all visible text content",
            "visible_elements": ["list", "of", "all", "visible", "elements"],
            "context": "detailed description of current context",
            "urls": ["any", "visible", "urls"],
            "files": ["any", "visible", "file", "names"]
        }"""
        
        response = model.generate_content([prompt, *image_parts])
        analysis = response.text
        
        # Clean up
        os.remove(temp_path)
        
        # Try to parse the response as JSON
        try:
            return json.loads(analysis)
        except json.JSONDecodeError:
            # If JSON parsing fails, return a formatted string
            return {
                "current_app": "Unknown",
                "visible_text": analysis,
                "visible_elements": [],
                "context": "Unable to parse screen analysis",
                "urls": [],
                "files": []
            }
            
    except Exception as e:
        print(f"Error analyzing screen: {e}")
        return {
            "current_app": "Unknown",
            "visible_text": "Error analyzing screen",
            "visible_elements": [],
            "context": str(e),
            "urls": [],
            "files": []
        }

def analyze_code_context():
    """Analyze the current code context if in an editor."""
    try:
        screen = capture_screen()
        if screen is None:
            return None
        
        # Save screen for analysis
        temp_path = os.path.join(tempfile.gettempdir(), "code_analysis.png")
        cv2.imwrite(temp_path, screen)
        
        # Use AI to analyze the code
        model = genai.GenerativeModel('gemini-1.5-flash')
        image_parts = [
            {
                "mime_type": "image/png",
                "data": open(temp_path, "rb").read()
            }
        ]
        
        prompt = """You are a code analysis expert. Analyze this code screenshot and provide detailed information about:
        1. The programming language being used
        2. The complete code visible in the editor
        3. Any syntax errors, runtime errors, or warnings
        4. The file name and path if visible
        5. Any error messages or stack traces
        6. The current cursor position and selected text
        
        Be very thorough in your analysis. Look for:
        - Syntax errors (missing brackets, semicolons, etc.)
        - Runtime errors (undefined variables, type mismatches, etc.)
        - Logical errors (incorrect logic, infinite loops, etc.)
        - Style issues (indentation, naming conventions, etc.)
        
        Respond in this exact JSON format:
        {
            "language": "programming language",
            "code": "complete code content",
            "errors": [
                {
                    "type": "error type (syntax/runtime/warning)",
                    "message": "detailed error message",
                    "line": "line number",
                    "suggestion": "detailed fix suggestion"
                }
            ],
            "file_info": {
                "name": "file name",
                "path": "file path"
            },
            "cursor": {
                "line": "current line",
                "column": "current column",
                "selection": "selected text if any"
            }
        }"""
        
        response = model.generate_content([prompt, *image_parts])
        analysis = response.text
        
        # Clean up
        os.remove(temp_path)
        
        try:
            return json.loads(analysis)
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract useful information from the text
            return {
                "language": "Unknown",
                "code": analysis,
                "errors": [{
                    "type": "unknown",
                    "message": "Unable to parse code analysis",
                    "line": "unknown",
                    "suggestion": "Please try again or provide more context"
                }],
                "file_info": {
                    "name": "unknown",
                    "path": "unknown"
                },
                "cursor": {
                    "line": "unknown",
                    "column": "unknown",
                    "selection": ""
                }
            }
            
    except Exception as e:
        print(f"Error analyzing code: {e}")
        return None

def fix_code_errors(code_analysis):
    """Fix code errors based on the analysis."""
    try:
        if not code_analysis or not code_analysis.get("errors"):
            return None
            
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Create a detailed prompt for fixing the errors
        prompt = f"""You are an expert code fixer. Fix the following code errors:
        
        Language: {code_analysis.get('language', 'Unknown')}
        Current code:
        {code_analysis.get('code', '')}
        
        Errors to fix:
        {json.dumps(code_analysis.get('errors', []), indent=2)}
        
        Requirements:
        1. Fix ALL errors identified in the analysis
        2. Maintain the original code structure and style
        3. Add any necessary imports or dependencies
        4. Ensure the code is complete and runnable
        5. Add comments explaining major changes
        
        Provide the complete fixed code that addresses all errors.
        Only provide the code, no explanations."""
        
        response = model.generate_content(prompt)
        fixed_code = response.text.strip()
        
        # Clean up the response
        if fixed_code.startswith('```'):
            fixed_code = fixed_code.split('\n', 1)[1]
        if fixed_code.endswith('```'):
            fixed_code = fixed_code.rsplit('\n', 1)[0]
            
        return fixed_code.strip()
        
    except Exception as e:
        print(f"Error fixing code: {e}")
        return None

def process_command(command: str) -> str:
    """Process user commands using AI to understand and execute the request."""
    try:
        # First, analyze the current screen context
        screen_analysis = analyze_screen()
        
        # Check if this is a screen analysis request
        if any(phrase in command.lower() for phrase in ["what do you see", "what's on my screen", "analyze screen", "what's visible"]):
            response = f"I can see:\n"
            if screen_analysis.get("current_app"):
                response += f"- You're currently in: {screen_analysis['current_app']}\n"
            if screen_analysis.get("visible_text"):
                response += f"- Visible text: {screen_analysis['visible_text']}\n"
            if screen_analysis.get("visible_elements"):
                response += f"- Visible elements: {', '.join(screen_analysis['visible_elements'])}\n"
            if screen_analysis.get("context"):
                response += f"- Context: {screen_analysis['context']}\n"
            if screen_analysis.get("urls"):
                response += f"- URLs: {', '.join(screen_analysis['urls'])}\n"
            if screen_analysis.get("files"):
                response += f"- Files: {', '.join(screen_analysis['files'])}\n"
            return response
        
        # Check if we're in a code editor and there are errors
        if "vscode" in screen_analysis.get("current_app", "").lower():
            code_analysis = analyze_code_context()
            if code_analysis:
                # If user is asking about errors or to fix them
                if any(phrase in command.lower() for phrase in ["fix", "error", "bug", "issue", "problem", "wrong"]):
                    fixed_code = fix_code_errors(code_analysis)
                    if fixed_code:
                        # Write the fixed code back to the file
                        file_path = code_analysis.get("file_info", {}).get("path")
                        if file_path and os.path.exists(file_path):
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(fixed_code)
                            return f"I've fixed the errors in your code. The changes have been saved to {file_path}"
                        else:
                            # Try to find the file in the current directory
                            file_name = code_analysis.get("file_info", {}).get("name")
                            if file_name:
                                current_dir = os.getcwd()
                                possible_path = os.path.join(current_dir, file_name)
                                if os.path.exists(possible_path):
                                    with open(possible_path, 'w', encoding='utf-8') as f:
                                        f.write(fixed_code)
                                    return f"I've fixed the errors in your code. The changes have been saved to {possible_path}"
                            return "I found some errors in your code, but I couldn't save the fixes. Could you please save the file first?"
                    else:
                        return "I found some errors in your code, but I'm having trouble fixing them. Could you share more details about the errors?"
                else:
                    # Just report the errors
                    error_list = "\n".join([f"- {e.get('type')} on line {e.get('line')}: {e.get('message')}\n  Suggestion: {e.get('suggestion')}" 
                                          for e in code_analysis.get("errors", [])])
                    return f"I see some issues in your code:\n{error_list}\n\nWould you like me to fix these errors?"
        
        # Use AI to understand the user's intent with better context
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        intent_prompt = f"""Analyze this user request and understand their intent: "{command}"

        Current screen context: {screen_analysis}

        Break down what the user wants to do and provide specific details.
        Consider the current screen context when determining the action.
        
        Examples of how to handle different requests:
        1. For opening websites:
           - "open youtube.com" -> Open the website in browser
           - "search for pokemon" -> Open browser and search
           - "go to google" -> Open Google in browser
        
        2. For opening applications:
           - "open notepad" -> Open Notepad
           - "open chrome" -> Open Chrome browser
           - "open this app" -> Use screen context to identify and open the visible app
        
        3. For YouTube/music (ONLY if explicitly mentioned):
           - "play despacito on youtube" -> Open YouTube and play the video
           - "play pokemon episode 1 on youtube" -> Search and play the specific video
           - "play some music on youtube" -> Open YouTube and play music
           - "search for despacito on youtube" -> Search YouTube for the video
        
        4. For code/writing:
           - "write a story" -> Create and open a text file
           - "create python code" -> Generate and run Python code
           - "make an html page" -> Create and open HTML file
        
        5. For system info:
           - "show system info" -> Display system information
           - "what's my cpu usage" -> Show CPU usage
        
        Respond with a JSON object containing:
        {{
            "intent": "what the user wants to do",
            "steps": [
                {{
                    "action": "specific action to take",
                    "details": "specific details for this action",
                    "parameters": {{"param1": "value1"}}
                }}
            ],
            "context": "any additional context or requirements"
        }}"""
        
        intent_response = model.generate_content(intent_prompt)
        intent_plan = intent_response.text.strip()
        
        try:
            import json
            plan = json.loads(intent_plan)
            steps = plan.get("steps", [])
            context = plan.get("context", "")
            
            results = []
            for step in steps:
                action = step.get("action", "").lower()
                details = step.get("details", "")
                params = step.get("parameters", {})
                
                # Handle website/browser actions
                if "website" in action or "browser" in action or "search" in action:
                    url = params.get("url", details)
                    if not url.startswith(('http://', 'https://')):
                        if "youtube" in url.lower() or "youtube" in command.lower():
                            results.append(play_youtube_video(url))
                        else:
                            results.append(open_browser(url))
                
                # Handle application opening
                elif "open" in action or "launch" in action:
                    app_name = params.get("app", details)
                    if app_name:
                        if "this" in app_name.lower() or "that" in app_name.lower():
                            # Use screen context to identify the app
                            screen_data = json.loads(screen_analysis)
                            visible_elements = screen_data.get("visible_elements", [])
                            if visible_elements:
                                app_name = visible_elements[0]  # Use the first visible element
                        subprocess.Popen(["start", app_name])
                        results.append(f"I've opened {app_name} for you")
                
                # Handle YouTube/music requests (only if explicitly mentioned)
                elif ("youtube" in action or "play" in action or "music" in action) and "youtube" in command.lower():
                    search_query = params.get("query", details)
                    if not search_query:
                        search_query = command.replace("play", "").replace("youtube", "").strip()
                    results.append(play_youtube_video(search_query))
                
                # Handle code generation
                elif "code" in action or "program" in action:
                    language = params.get("language", "Python")
                    task = params.get("task", details)
                    code = generate_code(task, language)
                    if code:
                        extension = ".py" if language.lower() == "python" else ".html"
                        file_path = create_and_write_file(code, extension)
                        if file_path:
                            open_in_editor(file_path)
                            if language.lower() == "python":
                                output = run_python_file(file_path)
                                results.append(f"I've created and run the Python code. Output:\n{output}")
                            else:
                                open_html_in_browser(file_path)
                                results.append(f"I've created the {language} code and opened it in your browser")
                
                # Handle writing/story requests
                elif "write" in action or "story" in action:
                    content_prompt = f"""Write {details} or a story about {details}.
                    Make it engaging and well-written."""
                    content_response = model.generate_content(content_prompt)
                    content = content_response.text.strip()
                    
                    file_path = create_and_write_file(content, ".txt")
                    if file_path:
                        open_in_editor(file_path)
                        results.append(f"I've created the content and opened it in Notepad")
                
                # Handle system information
                elif "system" in action or "info" in action:
                    results.append(get_system_info())
            
            return "\n".join(results) if results else f"I understand you want to {plan.get('intent', 'help')}. {context}"
            
        except json.JSONDecodeError:
            # If JSON parsing fails, try direct command understanding
            direct_prompt = f"""What does the user want to do with this command: "{command}"?
            Current screen context: {screen_analysis}
            Provide a simple, direct action to take."""
            direct_response = model.generate_content(direct_prompt)
            action = direct_response.text.strip().lower()
            
            # Handle direct actions based on keywords
            if ("youtube" in action or "play" in action) and "youtube" in command.lower():
                return play_youtube_video(command.replace("play", "").replace("youtube", "").strip())
            elif "open" in action:
                app_name = command.replace("open", "").strip()
                subprocess.Popen(["start", app_name])
                return f"I've opened {app_name} for you"
            elif "search" in action:
                return open_browser(f"https://www.google.com/search?q={command.replace('search', '').strip()}")
            elif "write" in action or "story" in action:
                content_prompt = f"""Write content based on this request: "{command}"
                Make it engaging and well-written."""
                content_response = model.generate_content(content_prompt)
                content = content_response.text.strip()
                
                file_path = create_and_write_file(content, ".txt")
                if file_path:
                    open_in_editor(file_path)
                    return "I've created the content and opened it in Notepad"
            elif "system" in action or "info" in action:
                return get_system_info()
            else:
                return "I understand you want me to help. Could you please rephrase your request?"
        
    except Exception as e:
        print(f"Error processing command: {e}")
        return "I'm here to help. What would you like me to do?"

# Create the main agent
root_agent = Agent(
    name="SystemControlAssistant",
    model="gemini-2.0-flash-lite",
    instruction=(
        "You are a helpful system control assistant that can execute commands and control the computer. "
        "You can:\n"
        "1. Open browsers and websites\n"
        "2. Show system information\n"
        "3. Run system commands\n"
        "4. Open applications\n"
        "5. Generate and write code in various languages\n"
        "6. Execute code and show results\n\n"
        "Always be friendly and natural in your responses. Don't mention errors or technical details to the user. "
        "Just confirm that you've completed their request in a conversational way."
    ),
    description="An assistant that can control your system and execute commands."
)

# Initialize session service
session_service = InMemorySessionService()
APP_NAME = "SystemControlApp"
USER_ID = "default_user"

# Global session variable
session = None

async def initialize_session():
    """Initialize the session asynchronously."""
    global session
    try:
        session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
        if session is None:
            raise Exception("Failed to create session")
        return session
    except Exception as e:
        print(f"Error creating session: {e}")
        raise

async def process_user_input(user_input: str) -> str:
    """Process user input and return agent response."""
    try:
        # First try to process as a system command using AI
        command_response = process_command(user_input)
        if command_response and not command_response.startswith("I understand you want me to help"):
            return command_response
        
        # If not a system command, use the LLM for general conversation
        global session
        if session is None:
            await initialize_session()
        
        user_message = genai_types.Content(role='user', parts=[genai_types.Part(text=user_input)])
        
        from google.adk.runners import Runner
        runner = Runner(
            agent=root_agent,
            app_name=APP_NAME,
            session_service=session_service
        )
        
        agent_reply = ""
        async for event in runner.run_async(user_id=USER_ID, session_id=session.id, new_message=user_message):
            if hasattr(event, 'type'):
                if event.type == 'final_response':
                    if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                agent_reply = part.text
                elif event.type == 'error':
                    error_text = event.content.parts[0].text if hasattr(event, 'content') and event.content and hasattr(event.content, 'parts') else 'Unknown error'
                    print(f"Error: {error_text}")
        
        return agent_reply if agent_reply else "I understand you want me to help. Could you please rephrase your request?"
    
    except Exception as e:
        print(f"Error processing input: {e}")
        return "I'm here to help. What would you like me to do?" 