from openai import OpenAI

class TinyAgent:

    def __init__(self, model, tokenizer=None, debug=False, api_key=None):

        self.model = model
        self.messages = []
        self.max_tokens = 10072
        self.debug = False
        self.reasoning_effort = "low"
        self.temperature = 1
        self.port = 11434
        self.client = OpenAI(api_key=api_key)

    def clear_messages(self):
        self.messages = list()

    def add_message(self,message_type, message):
        self.messages.append({"role": message_type, "content":message})

    def add_system_message(self, message):
        self.add_message("system", message)

    def add_user_message(self, message):
        self.add_message("user", message)

    def add_assistant_message(self, message):
        self.add_message("assistant", message)

    def set_max_tokens(self, max_tokens):
        self.max_tokens = max_tokens

    def set_debug(self, debug):
        self.debug = debug

    def set_reasoning_effort(self, reasoning_effort):
        self.reasoning_effort = reasoning_effort
        
    # Added
    def set_port(self, port):
        self.port = port
        
    # Added
    def set_client(self, api_key):
        self.port = port

    def call(self, prompt="", response_type="text", cache=True):
        messages = self.messages.copy()
        if prompt:
            messages.append({"role": "user", "content":prompt})
        if cache:
            self.add_user_message(prompt)

        if "gpt-5" in self.model:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                reasoning={"effort": self.reasoning_effort},
                text={
                    "format": {
                      "type": response_type
                    },
                    "verbosity": "low"
                  },
            )
            reply = response.output_text

        elif "gpt-4" in self.model or "o3" in self.model or "o4" in self.model:
            response = self.client.responses.create(
              model=self.model,
              input=messages,
              temperature=self.temperature,
              max_output_tokens=self.max_tokens,
              top_p=1,
              text={
                "format": {
                  "type": response_type # "text", "json_object"
                }
              }
            )
            reply = response.output_text
            
        #Added    
        elif "gpt-oss:20b" in self.model or "gpt-oss:120b" in self.model:
            
            client = OpenAI(
                base_url=f"http://localhost:{self.port}/v1", # Local Ollama API
                api_key="ollama" # Dummy key
            )
            
            messages.insert(0, {"role": "system", "content": f"Reasoning: {self.reasoning_effort}"})
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            
            reply = response.choices[0].message.content
        
        #Added    
        elif "deepseek-r1:70b" in self.model:
            
            client = OpenAI(
                base_url=f"http://localhost:{self.port}/v1", # Local Ollama API
                api_key="ollama" # Dummy key
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
            
            reply = response.choices[0].message.content
            reply = re.sub(r"<think>.*?</think>\n\n", "", reply, flags=re.DOTALL)
            
        if self.debug:
            print(reply)
        if cache:
            self.add_assistant_message(reply)
        return reply

    def load_json(self,s):
        import json
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    def call_json(self, prompt=""):
        self.add_system_message("Reply must be JSON format.")
        reply = self.call(prompt=prompt, response_type="json_object")
        if not reply:
            print("Empty reply")
            return None

        reply = reply.strip()
        if reply.startswith("```json"):
            reply = reply[len("```json"):].strip()
            if reply.endswith("```"):
                reply = reply[:-3].strip()

        # Use OR, and guard length
        if not (reply.startswith("{") and reply.endswith("}")):
            print("Not JSON structure")
            return None

        try:
            return self.load_json(reply)
        except Exception:
            print("Error parsing JSON")
            return None