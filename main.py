"""
AI Suite for Startup Workflow Optimization
-----------------------------------------
A hierarchical AI system with shared memory that lives in a single folder.
"""

import os
import json
import datetime
import logging
import uuid
import time
from typing import Dict, List, Any
from enum import Enum, auto

# Configuration
CONFIG = {
    "api_keys": {
        "gemini": os.environ.get("GEMINI_API_KEY", ""),
        "claude": os.environ.get("CLAUDE_API_KEY", ""),
    },
    "memory_path": "memory/shared_memory.json",
    "log_path": "logs/system_log.txt",
}


# Define Role Types
class RoleType(Enum):
    MANAGER = auto()
    BRAND_ADVISOR = auto()
    DEV_LOG_PROCESSOR = auto()
    POST_CREATOR = auto()
    RESEARCHER = auto()
    PITCH_GENERATOR = auto()
    SOCIAL_HANDLER = auto()


# Define Message Types
class MessageType(Enum):
    COMMAND = auto()  # CEO command
    RESPONSE = auto()  # Node response
    NOTIFICATION = auto()  # System notification
    ERROR = auto()  # Error message


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(CONFIG["log_path"]), logging.StreamHandler()],
)
logger = logging.getLogger("ai-suite")


class SharedMemory:
    """Central shared memory system for all AI nodes"""

    def __init__(self, memory_path: str):
        self.memory_path = memory_path
        self.memory = self._load_memory()
        self.session_context = {}

    def _load_memory(self) -> Dict:
        """Load memory from disk"""
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)

        if os.path.exists(self.memory_path):
            try:
                with open(self.memory_path, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to load memory file. Creating new memory.")
                return self._initialize_memory()
        else:
            return self._initialize_memory()

    def _initialize_memory(self) -> Dict:
        """Create a new memory structure"""
        memory = {
            "projects": {},
            "goals": {},
            "tasks": {},
            "brand_info": {},
            "research": {},
            "posts": {},
            "pitches": {},
            "social_interactions": {},
            "history": [],
        }
        self._save_memory(memory)
        return memory

    def _save_memory(self, memory: Dict | None = None) -> None:
        """Save memory to disk"""
        if memory is None:
            memory = self.memory

        with open(self.memory_path, "w") as f:
            json.dump(memory, f, indent=2)

    def add_history_entry(self, action: str, details: Any) -> None:
        """Add an entry to the history log"""
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action,
            "details": details,
        }
        self.memory["history"].append(entry)
        self._save_memory()

    def get(self, key_path: str) -> Any:
        """Get a value from memory using dot notation"""
        keys = key_path.split(".")
        current = self.memory

        for key in keys:
            if key in current:
                current = current[key]
            else:
                return None

        return current

    def set(self, key_path: str, value: Any) -> None:
        """Set a value in memory using dot notation"""
        keys = key_path.split(".")
        current = self.memory

        # Navigate to the nested location
        for _, key in enumerate(keys[:-1]):
            if key not in current:
                current[key] = {}
            current = current[key]

        # Set the value
        current[keys[-1]] = value
        self._save_memory()

    def set_session_context(self, key: str, value: Any) -> None:
        """Set a value in the temporary session context"""
        self.session_context[key] = value

    def get_session_context(self, key: str) -> Any:
        """Get a value from the temporary session context"""
        return self.session_context.get(key)


class AINode:
    """Base class for all AI nodes in the system"""

    def __init__(self, memory: SharedMemory, role: RoleType):
        self.memory = memory
        self.role = role
        self.node_id = str(uuid.uuid4())

    def process(self, instruction: str, context: Dict | None = None) -> Dict:
        """Process an instruction with the given context"""
        raise NotImplementedError("Each node must implement the process method")

    def call_ai_api(self, prompt: str, model: str = "gemini") -> str:
        """Call the appropriate AI API based on the model parameter"""
        if model == "gemini":
            return self._call_gemini_api(prompt)
        elif model == "claude":
            return self._call_claude_api(prompt)
        else:
            raise ValueError(f"Unsupported model: {model}")

    def _call_gemini_api(self, prompt: str) -> str:
        """Call the Gemini API"""
        # This is a simplified example - in a real implementation, use the official Gemini API client
        api_key = CONFIG["api_keys"]["gemini"]
        if not api_key:
            logger.error("Gemini API key not configured")
            return "ERROR: Gemini API key not configured"

        # Implement actual API call
        logger.info(f"Calling Gemini API with prompt: {prompt[:50]}...")
        try:
            # For the real implementation, you'd use Google's API client:
            # from google.generativeai import GenerativeModel
            # model = GenerativeModel('gemini-pro')
            # response = model.generate_content(prompt)
            # return response.text

            # Placeholder for development
            time.sleep(1)  # Simulate API latency
            return f"Detailed response to: {prompt[:50]}..."
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return f"ERROR: API call failed: {str(e)}"

    def _call_claude_api(self, prompt: str) -> str:
        """Call the Claude API"""
        # This is a simplified example - in a real implementation, use the official Claude API client
        api_key = CONFIG["api_keys"]["claude"]
        if not api_key:
            logger.error("Claude API key not configured")
            return "ERROR: Claude API key not configured"

        # Implement actual API call
        logger.info(f"Calling Claude API with prompt: {prompt[:50]}...")
        try:
            # For the real implementation, you'd use Anthropic's API client:
            # from anthropic import Anthropic
            # client = Anthropic(api_key=api_key)
            # message = client.messages.create(
            #     model="claude-3-sonnet-20240229",
            #     max_tokens=1000,
            #     system="You are a helpful AI assistant.",
            #     messages=[{"role": "user", "content": prompt}]
            # )
            # return message.content[0].text

            # Placeholder for development
            time.sleep(1.5)  # Simulate API latency
            return f"Thoughtful analysis regarding: {prompt[:50]}..."
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return f"ERROR: API call failed: {str(e)}"


class ManagerNode(AINode):
    """Central manager that coordinates all other AI nodes"""

    def __init__(self, memory: SharedMemory):
        super().__init__(memory, RoleType.MANAGER)
        self.nodes = {}

    def register_node(self, role: RoleType, node: AINode) -> None:
        """Register a specialized node with the manager"""
        self.nodes[role] = node

    def process(self, instruction: str, context: Dict | None = None) -> Dict:
        """Process CEO instructions and delegate to appropriate nodes"""
        if context is None:
            context = {}

        # Log the instruction
        logger.info(f"Manager received instruction: {instruction}")
        self.memory.add_history_entry(
            "manager_instruction", {"instruction": instruction}
        )

        # Determine which node(s) should handle this instruction
        handling_plan = self._create_handling_plan(instruction)

        # Execute the plan
        results = self._execute_plan(handling_plan, instruction, context)

        return {
            "status": "completed",
            "original_instruction": instruction,
            "handling_plan": handling_plan,
            "results": results,
        }

    def _create_handling_plan(self, instruction: str) -> Dict:
        """Determine how to handle an instruction"""
        # This would be a more sophisticated AI-based routing in a real implementation
        plan = {
            "primary_role": None,
            "supporting_roles": [],
            "execution_order": [],
            "instruction_breakdown": {},
        }

        # Use AI to determine routing - this is a key part of maintaining hierarchy
        routing_prompt = f"""
        Analyze this instruction and determine which specialized AI node should handle it:
        
        INSTRUCTION: {instruction}
        
        Available nodes:
        - BRAND_ADVISOR: Brand identity, colors, taglines, visual design
        - DEV_LOG_PROCESSOR: Converting development logs to posts
        - POST_CREATOR: Creating and scheduling final posts
        - RESEARCHER: Market or technical research
        - PITCH_GENERATOR: Sales pitches, demos, presentations
        - SOCIAL_HANDLER: Social media interaction management
        
        Respond with the primary node name and any supporting nodes needed.
        """

        # For demo purposes, we'll use a simplified keyword approach
        # In production, this would call an AI to make this decision
        lower_instruction = instruction.lower()

        if (
            "brand" in lower_instruction
            or "color" in lower_instruction
            or "tagline" in lower_instruction
        ):
            plan["primary_role"] = RoleType.BRAND_ADVISOR

        elif "dev log" in lower_instruction or "development log" in lower_instruction:
            plan["primary_role"] = RoleType.DEV_LOG_PROCESSOR

        elif "post" in lower_instruction or "schedule" in lower_instruction:
            plan["primary_role"] = RoleType.POST_CREATOR

        elif "research" in lower_instruction or "market" in lower_instruction:
            plan["primary_role"] = RoleType.RESEARCHER

        elif "pitch" in lower_instruction or "sell" in lower_instruction:
            plan["primary_role"] = RoleType.PITCH_GENERATOR

        elif any(
            platform in lower_instruction
            for platform in ["reddit", "discord", "twitter", "linkedin"]
        ):
            plan["primary_role"] = RoleType.SOCIAL_HANDLER

        else:
            # Default to manager handling it directly
            plan["primary_role"] = RoleType.MANAGER

        # Add supporting roles if needed
        if (
            "research" in lower_instruction
            and plan["primary_role"] != RoleType.RESEARCHER
        ):
            plan["supporting_roles"].append(RoleType.RESEARCHER)

        # If we're creating a pitch, we might want brand info
        if plan["primary_role"] == RoleType.PITCH_GENERATOR:
            plan["supporting_roles"].append(RoleType.BRAND_ADVISOR)

        # If posting to social, we might want the post creator involved
        if (
            plan["primary_role"] == RoleType.SOCIAL_HANDLER
            and "post" in lower_instruction
        ):
            plan["supporting_roles"].append(RoleType.POST_CREATOR)

        # Create execution order - supporting roles come after the primary
        plan["execution_order"] = [plan["primary_role"]] + plan["supporting_roles"]

        # Break down the instruction for each role
        for role in plan["execution_order"]:
            plan["instruction_breakdown"][
                role.name
            ] = f"Handle this as a {role.name}: {instruction}"

        return plan

    def _execute_plan(self, plan: Dict, instruction: str, context: Dict) -> Dict:
        """Execute the handling plan by delegating to appropriate nodes"""
        results = {}

        for role in plan["execution_order"]:
            if role in self.nodes:
                node_context = context.copy()
                # Add previous results to context for chaining
                node_context.update({"previous_results": results})

                # Execute the node
                node_result = self.nodes[role].process(instruction, node_context)
                results[role.name] = node_result

                # Update session context with results
                self.memory.set_session_context(
                    f"last_{role.name.lower()}_result", node_result
                )
            else:
                logger.warning(f"No node registered for role {role}")

        return results


class BrandAdvisorNode(AINode):
    """Specialized node for brand advisory"""

    def __init__(self, memory: SharedMemory):
        super().__init__(memory, RoleType.BRAND_ADVISOR)

    def process(self, instruction: str, context: Dict | None = None) -> Dict:
        if context is None:
            context = {}

        # Create a specialized prompt for brand advisory
        prompt = f"""
        You are a brand advisory AI. 
        Please analyze this instruction and provide brand guidance:
        
        INSTRUCTION: {instruction}
        
        Provide the following:
        1. Color palette recommendations
        2. Tagline suggestions
        3. Brand voice characteristics
        4. Visual identity elements
        """

        # Call AI API
        response = self.call_ai_api(prompt)

        # Process and structure the response
        # In a real implementation, this would parse the AI's response into structured data
        result = {
            "colors": ["#3A86FF", "#FF006E", "#FFBE0B"],  # Example data
            "taglines": ["Innovate with Purpose", "Tomorrow's Solutions Today"],
            "brand_voice": "Professional, innovative, approachable",
            "raw_response": response,
        }

        # Save to shared memory
        project_name = context.get("project_name", "default_project")
        self.memory.set(f"brand_info.{project_name}", result)

        return result


class DevLogProcessorNode(AINode):
    """Specialized node for processing development logs into posts"""

    def __init__(self, memory: SharedMemory):
        super().__init__(memory, RoleType.DEV_LOG_PROCESSOR)

    def process(self, instruction: str, context: Dict | None = None) -> Dict:
        if context is None:
            context = {}

        # Extract dev log from context or file system
        dev_log = context.get("dev_log", "")
        if not dev_log and "log_file" in context:
            try:
                with open(context["log_file"], "r") as f:
                    dev_log = f.read()
            except Exception as e:
                logger.error(f"Failed to read dev log file: {e}")
                return {"error": f"Failed to read dev log file: {str(e)}"}

        # If no log file specified, look for logs in the current directory
        if not dev_log and "log_file" not in context:
            dev_log = self._find_latest_dev_log()

        if not dev_log:
            return {
                "error": "No development log found. Please provide a log file or content."
            }

        # Get brand info for consistency
        project_name = context.get("project_name", "default_project")
        brand_info = self.memory.get(f"brand_info.{project_name}")

        # Create specialized prompt
        prompt = f"""
        You are a technical content creator AI.
        Transform this development log into an engaging social media post:
        
        DEV LOG:
        {dev_log}
        
        {"BRAND VOICE: " + brand_info.get("brand_voice", "") if brand_info else ""}
        
        Create a post that:
        1. Highlights key achievements and milestones
        2. Mentions challenges overcome (if any)
        3. Shows progress and momentum
        4. Generates interest in the project
        5. Ends with a compelling call to action
        """

        # Call AI API
        response = self.call_ai_api(prompt)

        # Process the response
        key_achievements = self._extract_achievements(dev_log)

        result = {
            "post_draft": response,
            "word_count": len(response.split()),
            "key_achievements": key_achievements,
            "suggested_hashtags": self._generate_hashtags(response, key_achievements),
            "dev_log_summary": self._summarize_dev_log(dev_log),
        }

        # Save to shared memory
        timestamp = datetime.datetime.now().isoformat()
        self.memory.set(f"posts.dev_log_posts.{timestamp}", result)

        return result

    def _find_latest_dev_log(self) -> str:
        """Find the latest development log in the current directory"""
        # This would search for dev log files in the current directory
        # For demonstration, we'll return a mock log
        return """
        2025-05-03 - Development Log
        
        - Implemented user authentication system
        - Fixed responsive design issues on mobile
        - Added caching layer to improve API performance
        - Started work on analytics dashboard
        - Currently blocked on third-party API integration
        """

    def _extract_achievements(self, dev_log: str) -> List[str]:
        """Extract key achievements from the dev log"""
        # In a real implementation, this would use NLP to find achievements
        # For demonstration, we'll do basic line extraction
        achievements = []
        for line in dev_log.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("*"):
                # Extract bullet points as achievements
                achievement = line[1:].strip()
                if (
                    achievement
                    and "not " not in achievement.lower()
                    and "issue" not in achievement.lower()
                ):
                    achievements.append(achievement)

        return achievements[:3]  # Return top 3 achievements

    def _generate_hashtags(self, post: str, achievements: List[str]) -> List[str]:
        """Generate relevant hashtags based on the post content"""
        # Standard hashtags
        hashtags = ["#BuildInPublic", "#StartupJourney", "#DevLife"]

        # Add specific hashtags based on content
        lower_post = post.lower()

        if "api" in lower_post or any("api" in a.lower() for a in achievements):
            hashtags.append("#API")

        if "user" in lower_post or "auth" in lower_post:
            hashtags.append("#UserExperience")

        if "performance" in lower_post or "speed" in lower_post:
            hashtags.append("#Performance")

        if "design" in lower_post or "ui" in lower_post or "ux" in lower_post:
            hashtags.append("#Design")

        return hashtags

    def _summarize_dev_log(self, dev_log: str) -> str:
        """Create a brief summary of the development log"""
        # In a real implementation, this would use AI to summarize
        # For demonstration, we'll return a simple summary
        return "Summary: Development progress with focus on authentication, performance and UI improvements."


class PostCreatorNode(AINode):
    """Specialized node for creating and scheduling posts"""

    def __init__(self, memory: SharedMemory):
        super().__init__(memory, RoleType.POST_CREATOR)

    def process(self, instruction: str, context: Dict | None = None) -> Dict:
        if context is None:
            context = {}

        # Get the post draft if it came from DevLogProcessor
        post_draft = ""
        if (
            "previous_results" in context
            and "DEV_LOG_PROCESSOR" in context["previous_results"]
        ):
            post_draft = context["previous_results"]["DEV_LOG_PROCESSOR"].get(
                "post_draft", ""
            )

        # Get brand info from memory for consistency
        project_name = context.get("project_name", "default_project")
        brand_info = self.memory.get(f"brand_info.{project_name}")

        # Determine target platforms
        platforms = self._determine_platforms(instruction)

        # Create specialized prompt
        prompt = f"""
        You are a social media content specialist AI.
        
        INSTRUCTION: {instruction}
        
        POST DRAFT: {post_draft}
        
        {"BRAND GUIDELINES: " + json.dumps(brand_info) if brand_info else ""}
        
        TARGET PLATFORMS: {', '.join(platforms)}
        
        Finalize this post for publishing. Consider:
        1. Platform-specific formatting for {', '.join(platforms)}
        2. Optimal posting time
        3. Engagement hooks
        4. Call to action
        5. Brand voice consistency
        """

        # Call AI API
        response = self.call_ai_api(prompt)

        # Process the response
        result = {
            "final_post": response,
            "platform_versions": self._create_platform_versions(response, platforms),
            "suggested_posting_time": self._determine_optimal_posting_time(platforms),
            "platforms": platforms,
            "brand_alignment": "High" if brand_info else "Unknown",
            "needs_approval": True,
            "approval_status": "pending",
        }

        # Save to shared memory
        timestamp = datetime.datetime.now().isoformat()
        self.memory.set(f"posts.final_posts.{timestamp}", result)

        return result

    def _determine_platforms(self, instruction: str) -> List[str]:
        """Determine which platforms to target based on instruction"""
        platforms = ["Twitter", "LinkedIn"]  # Default platforms

        lower_instruction = instruction.lower()
        if "twitter" in lower_instruction or "x" in lower_instruction:
            platforms = ["Twitter"]
        elif "linkedin" in lower_instruction:
            platforms = ["LinkedIn"]
        elif "reddit" in lower_instruction:
            platforms = ["Reddit"]
        elif "facebook" in lower_instruction:
            platforms = ["Facebook"]
        elif "instagram" in lower_instruction:
            platforms = ["Instagram"]

        return platforms

    def _create_platform_versions(
        self, response: str, platforms: List[str]
    ) -> Dict[str, str]:
        """Create platform-specific versions of the post"""
        # In a real implementation, you would modify the post for each platform
        # This is a simplified version for demonstration
        versions = {}

        for platform in platforms:
            if platform == "Twitter":
                # Twitter has character limits
                versions[platform] = response[:280]
            elif platform == "LinkedIn":
                # LinkedIn can be more professional
                versions[platform] = f"#ProfessionalUpdate\n\n{response}"
            elif platform == "Reddit":
                # Reddit might need more detail
                versions[platform] = (
                    f"{response}\n\nThoughts? Would love to hear your feedback!"
                )
            else:
                versions[platform] = response

        return versions

    def _determine_optimal_posting_time(self, platforms: List[str]) -> Dict[str, str]:
        """Determine the optimal posting time for each platform"""
        # In a real implementation, this would use analytics data
        # This is a simplified version for demonstration
        times = {}

        for platform in platforms:
            if platform == "Twitter":
                times[platform] = "12:00 PM EST"
            elif platform == "LinkedIn":
                times[platform] = "9:00 AM EST"
            elif platform == "Reddit":
                times[platform] = "7:00 PM EST"
            else:
                times[platform] = "10:00 AM EST"

        return times


class ResearcherNode(AINode):
    """Specialized node for market and technical research"""

    def __init__(self, memory: SharedMemory):
        super().__init__(memory, RoleType.RESEARCHER)

    def process(self, instruction: str, context: Dict | None = None) -> Dict:
        if context is None:
            context = {}

        # Determine research type
        research_type = "market"  # Default
        if "research_type" in context:
            research_type = context["research_type"]
        elif "technical" in instruction.lower():
            research_type = "technical"

        # Create specialized prompt
        prompt = f"""
        You are a specialized {research_type} research AI.
        
        RESEARCH QUERY: {instruction}
        
        Provide:
        1. Key findings summary
        2. Competitive landscape analysis
        3. Trends and opportunities
        4. Data-backed recommendations
        5. Sources for further investigation
        """

        # Call AI API with longer context allowed
        response = self.call_ai_api(prompt, model="gemini")

        # Process the response
        result = {
            "research_type": research_type,
            "findings": response,
            "timestamp": datetime.datetime.now().isoformat(),
            "query": instruction,
        }

        # Save to shared memory
        research_id = str(uuid.uuid4())[:8]
        self.memory.set(f"research.{research_type}.{research_id}", result)

        return result
