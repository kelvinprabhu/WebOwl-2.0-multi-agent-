# WebOwlMultiAgentRAG_Improved.py

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
import networkx as nx
from collections import defaultdict, deque
import time
from datetime import datetime

# Improved Web Owl personality configuration
WEB_OWL_PERSONALITY = {
    "humor_level": 0.2,
    "professionalism": 0.8,
    "helpfulness": 0.9,
    "curiosity": 0.7,
    "conversational": 0.8
}

class AgentRole(Enum):
    CONVERSATION_MANAGER = "conversation_manager"
    INFORMATION_STRUCTURER = "information_structurer"
    SITE_MAPPER = "site_mapper" 
    RESPONSE_STRUCTURER = "response_structurer"
    FINAL_VERIFIER = "final_verifier"

@dataclass
class WebOwlResponse:
    query: str
    final_answer: str
    structured_info: Dict[str, Any]
    site_navigation: Dict[str, Any]
    confidence_score: float
    sources_used: List[str]
    navigation_path: List[str]
    conversation_context: Optional[str] = None
    follow_up_suggestions: Optional[List[str]] = None
    humor_quote: Optional[str] = None

class ConversationManagerAgent:
    """Manages conversation context and user intent"""
    
    def __init__(self, llm_client, window_size: int = 5):
        self.llm = llm_client
        self.memory = ConversationBufferWindowMemory(
            k=window_size,
            return_messages=True
        )
        self.user_context = {}
        
    def analyze_user_intent(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Analyze user intent and conversation context"""
        
        # Get conversation history
        conversation_context = ""
        if session_id and session_id in self.user_context:
            recent_history = self.user_context[session_id].get('history', [])[-3:]
            conversation_context = "\n".join([
                f"Q: {item['query']}\nA: {item['answer'][:150]}..."
                for item in recent_history
            ])
        
        prompt = f"""
Analyze this user query in context of their conversation history:

Current Query: "{query}"

Recent Conversation Context:
{conversation_context or "No previous conversation"}

Determine:
1. Query type (factual, navigational, clarification, follow-up)
2. User intent (information seeking, problem solving, exploration)
3. Required response style (direct answer, detailed explanation, guidance)
4. Relationship to previous queries (continuation, new topic, clarification)
5. Suggested follow-up questions

Respond in JSON format with analysis.
"""
        
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
            
        # Fallback analysis
        return {
            "query_type": "factual",
            "user_intent": "information_seeking",
            "response_style": "detailed_explanation",
            "relationship": "new_topic" if not conversation_context else "continuation",
            "follow_up_suggestions": []
        }
    
    def update_conversation_context(self, session_id: str, query: str, response: str):
        """Update conversation context for a session"""
        if session_id not in self.user_context:
            self.user_context[session_id] = {
                'history': [],
                'preferences': {},
                'topics': set()
            }
        
        self.user_context[session_id]['history'].append({
            'query': query,
            'answer': response,
            'timestamp': datetime.now()
        })
        
        # Keep only recent history
        if len(self.user_context[session_id]['history']) > 10:
            self.user_context[session_id]['history'] = self.user_context[session_id]['history'][-10:]

# Improved WebOwlAgent with conversation awareness
class WebOwlAgent:
    """Enhanced base class for Web Owl agents with conversation awareness"""
    
    def __init__(self, role: AgentRole, llm_client, personality: Dict[str, float] = None):
        self.role = role
        self.llm = llm_client
        self.personality = personality or WEB_OWL_PERSONALITY
        self.memory = ConversationBufferWindowMemory(k=3, return_messages=True)
        
    def _add_personality_to_prompt(self, base_prompt: str, conversation_context: str = "") -> str:
        """Add Web Owl personality traits to prompts with conversation awareness"""
        personality_traits = f"""
You are Web Owl, an intelligent and conversational web navigation assistant with these traits:
- {int(self.personality['humor_level'] * 100)}% humor (light, appropriate humor)
- {int(self.personality['professionalism'] * 100)}% professionalism (accurate and reliable)
- {int(self.personality['helpfulness'] * 100)}% helpfulness (proactive assistance)
- {int(self.personality['conversational'] * 100)}% conversational (natural, flowing dialogue)

Your role: {self.role.value.replace('_', ' ').title()}

{f"Conversation Context: {conversation_context}" if conversation_context else ""}

{base_prompt}

Maintain continuity with previous interactions while providing fresh insights. Be conversational and engaging.
"""
        return personality_traits
    
    def generate_follow_up_suggestions(self, query: str, response_content: str) -> List[str]:
        """Generate contextual follow-up questions"""
        suggestions = [
            "Can you show me more details about this topic?",
            "How do I navigate to this information on the website?",
            "What related topics might interest me?",
            "Can you explain this in simpler terms?"
        ]
        return suggestions[:3]  # Return top 3

# Enhanced agents with conversation support
class ImprovedInformationStructurerAgent(WebOwlAgent):
    """Enhanced information structurer with conversation context"""
    
    def __init__(self, llm_client, retriever):
        super().__init__(AgentRole.INFORMATION_STRUCTURER, llm_client)
        self.retriever = retriever
        
    def structure_information(self, query: str, retrieved_chunks: List, 
                            conversation_context: str = "") -> Dict[str, Any]:
        """Structure information with conversation awareness"""
        
        prompt = self._add_personality_to_prompt(f"""
Analyze and structure the retrieved information for: "{query}"

Retrieved Information:
{self._format_chunks_for_analysis(retrieved_chunks)}

Consider the conversation context and:
1. Focus on information most relevant to user's apparent intent
2. Identify connections to previous topics discussed
3. Structure for clear, conversational delivery
4. Note any information that builds on previous responses

Provide structured JSON with:
- categorized_info: Organized by relevance and topic
- key_facts: Most important facts for this user
- contextual_connections: Links to previous conversation
- information_gaps: What might need clarification
- user_focused_insights: Insights tailored to user's journey
""", conversation_context)
        
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        try:
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
            
        return self._manual_structure_with_context(query, retrieved_chunks)
    
    def _format_chunks_for_analysis(self, chunks: List) -> str:
        """Format chunks with better context markers"""
        formatted = []
        for i, chunk in enumerate(chunks, 1):
            formatted.append(f"""
[Source {i}] ({getattr(chunk, 'score', 0):.3f} relevance)
From: {getattr(chunk, 'source_url', 'Unknown')}
Type: {getattr(chunk, 'source_type', 'Unknown')}
Content: {getattr(chunk, 'text', '')[:400]}...
""")
        return "\n".join(formatted)
    
    def _manual_structure_with_context(self, query: str, chunks: List) -> Dict[str, Any]:
        """Enhanced fallback structuring"""
        return {
            "categorized_info": {
                "primary_answer": [chunk.text for chunk in chunks[:2]],
                "supporting_details": [chunk.text for chunk in chunks[2:]]
            },
            "key_facts": [chunk.text[:150] for chunk in chunks[:3]],
            "contextual_connections": [],
            "information_gaps": ["May need additional context based on user needs"],
            "user_focused_insights": ["Direct answer provided with supporting information"]
        }

# Enhanced Response Structurer
class ImprovedResponseStructurerAgent(WebOwlAgent):
    """Creates conversational, user-friendly responses"""
    
    def __init__(self, llm_client):
        super().__init__(AgentRole.RESPONSE_STRUCTURER, llm_client)
        
    def structure_response(self, query: str, structured_info: Dict, 
                          navigation_analysis: Dict, user_intent: Dict = None,
                          conversation_context: str = "") -> Dict[str, Any]:
        """Create engaging, conversational response"""
        
        intent_info = user_intent or {"response_style": "detailed_explanation"}
        
        prompt = self._add_personality_to_prompt(f"""
Create a conversational, helpful response for: "{query}"

User Intent: {intent_info.get('response_style', 'detailed')} response needed
Query Type: {intent_info.get('query_type', 'factual')}

Information Available:
{json.dumps(structured_info, indent=2)}

Navigation Context:
{json.dumps(navigation_analysis, indent=2)}

Create a response that:
1. Directly answers the user's question conversationally
2. Provides context and background naturally
3. Includes clear next steps or navigation guidance
4. Suggests relevant follow-up topics
5. Maintains a helpful, engaging tone

Structure as natural conversation, not rigid sections.
""", conversation_context)
        
        messages = [SystemMessage(content=prompt)]
        response = self.llm.invoke(messages)
        
        # Generate follow-up suggestions
        follow_ups = self.generate_follow_up_suggestions(query, response.content)
        
        return {
            "structured_response": response.content,
            "confidence_indicators": self._assess_confidence(structured_info),
            "actionable_next_steps": self._generate_actionable_steps(navigation_analysis),
            "follow_up_suggestions": follow_ups,
            "response_type": intent_info.get('response_style', 'detailed')
        }
    
    def _assess_confidence(self, structured_info: Dict) -> Dict[str, Any]:
        """Enhanced confidence assessment"""
        key_facts_count = len(structured_info.get("key_facts", []))
        info_gaps_count = len(structured_info.get("information_gaps", []))
        
        return {
            "information_completeness": min(key_facts_count / 3.0, 1.0),
            "source_diversity": 0.8,  # Simplified for now
            "gap_analysis": max(1.0 - (info_gaps_count / 5.0), 0.0),
            "overall_confidence": min(key_facts_count / 3.0, 1.0)
        }
    
    def _generate_actionable_steps(self, navigation_analysis: Dict) -> List[str]:
        """Generate user-friendly action steps"""
        steps = []
        
        if navigation_analysis.get("navigation_paths"):
            steps.append("I can show you the exact path to navigate to this information")
        
        steps.append("Ask me for more details about any specific aspect")
        steps.append("I can help you explore related topics")
        
        return steps

class SiteMapper:
    """Enhanced site mapper that understands website structure and relationships"""
    
    def __init__(self, retriever):
        self.retriever = retriever
        self.site_graph = nx.DiGraph()
        self._build_site_graph()
        
    def _build_site_graph(self):
        """Build a comprehensive site graph from Neo4j data"""
        print("🦉 Building site navigation map...")
        
        with self.retriever.driver.session() as session:
            # Get all pages and their relationships
            result = session.run("""
                MATCH (p:Page)
                OPTIONAL MATCH (p)-[r:LINKS_TO]->(target:Page)
                OPTIONAL MATCH (p)-[:CONTAINS]->(a:Asset)
                RETURN p.url as page_url, p.text_len as content_length,
                       collect(DISTINCT {url: target.url, anchor: r.anchor_text}) as links,
                       collect(DISTINCT {url: a.url, type: a.type, filename: a.filename}) as assets
            """)
            
            for row in result:
                page_url = row['page_url']
                
                # Add page node with metadata
                self.site_graph.add_node(page_url, 
                                       node_type='page',
                                       content_length=row['content_length'],
                                       assets=row['assets'])
                
                # Add links as edges
                for link in row['links']:
                    if link['url']:
                        self.site_graph.add_edge(page_url, link['url'], 
                                               anchor_text=link['anchor'],
                                               edge_type='navigation')
                
                # Add assets as nodes and connections
                for asset in row['assets']:
                    if asset['url']:
                        asset_url = asset['url']
                        self.site_graph.add_node(asset_url,
                                               node_type='asset',
                                               asset_type=asset['type'],
                                               filename=asset['filename'])
                        self.site_graph.add_edge(page_url, asset_url,
                                               edge_type='contains')
        
        print(f"🗺️ Site map built: {self.site_graph.number_of_nodes()} pages/assets, {self.site_graph.number_of_edges()} connections")
    
    def find_navigation_path(self, start_url: str, target_content: str) -> List[str]:
        """Find the best navigation path to reach specific content"""
        
        # Find pages that might contain the target content
        relevant_pages = []
        with self.retriever.driver.session() as session:
            result = session.run("""
                MATCH (c:Chunk)
                WHERE c.text CONTAINS $content
                OPTIONAL MATCH (p:Page)-[:HAS_CHUNK]->(c)
                RETURN DISTINCT p.url as page_url
                LIMIT 5
            """, content=target_content.lower())
            
            relevant_pages = [row['page_url'] for row in result if row['page_url']]
        
        if not relevant_pages:
            return []
            
        # Find shortest path to most relevant page
        paths = []
        for target_page in relevant_pages:
            if start_url in self.site_graph and target_page in self.site_graph:
                try:
                    path = nx.shortest_path(self.site_graph, start_url, target_page)
                    paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return min(paths, key=len) if paths else []
    
    def get_page_context(self, url: str) -> Dict[str, Any]:
        """Get comprehensive context about a page"""
        if url not in self.site_graph:
            return {}
            
        node_data = self.site_graph.nodes[url]
        
        # Get incoming and outgoing links
        incoming = list(self.site_graph.predecessors(url))
        outgoing = list(self.site_graph.successors(url))
        
        # Get related assets
        assets = [n for n in outgoing if self.site_graph.nodes[n].get('node_type') == 'asset']
        
        return {
            'url': url,
            'content_length': node_data.get('content_length', 0),
            'incoming_links': len(incoming),
            'outgoing_links': len([n for n in outgoing if self.site_graph.nodes[n].get('node_type') == 'page']),
            'assets': assets,
            'parent_pages': incoming,
            'child_pages': [n for n in outgoing if self.site_graph.nodes[n].get('node_type') == 'page']
        }
    
    def generate_sitemap_summary(self) -> Dict[str, Any]:
        """Generate a summary of the site structure"""
        pages = [n for n in self.site_graph.nodes() if self.site_graph.nodes[n].get('node_type') == 'page']
        assets = [n for n in self.site_graph.nodes() if self.site_graph.nodes[n].get('node_type') == 'asset']
        
        # Find root pages (pages with no incoming links from other pages)
        root_pages = [n for n in pages if not any(
            self.site_graph.nodes[pred].get('node_type') == 'page' 
            for pred in self.site_graph.predecessors(n)
        )]
        
        # Calculate depth of each page
        page_depths = {}
        for root in root_pages:
            distances = nx.single_source_shortest_path_length(self.site_graph, root)
            for page, depth in distances.items():
                if self.site_graph.nodes[page].get('node_type') == 'page':
                    page_depths[page] = min(page_depths.get(page, float('inf')), depth)
        
        return {
            'total_pages': len(pages),
            'total_assets': len(assets),
            'root_pages': root_pages,
            'max_depth': max(page_depths.values()) if page_depths else 0,
            'avg_links_per_page': sum(len(list(self.site_graph.successors(p))) for p in pages) / len(pages) if pages else 0
        }

class SiteMappingAgent(WebOwlAgent):
    """Analyzes site structure and navigation patterns"""
    
    def __init__(self, llm_client, site_mapper: SiteMapper):
        super().__init__(AgentRole.SITE_MAPPER, llm_client)
        self.site_mapper = site_mapper
        
    def analyze_navigation(self, query: str, relevant_sources: List[str]) -> Dict[str, Any]:
        """Analyze navigation patterns and suggest optimal paths"""
        
        sitemap_summary = self.site_mapper.generate_sitemap_summary()
        
        # Get context for relevant sources
        source_contexts = {}
        for source_url in relevant_sources:
            source_contexts[source_url] = self.site_mapper.get_page_context(source_url)
        
        prompt = self._add_personality_to_prompt(f"""
Analyze the website navigation for query: "{query}"

Site Structure Summary:
{json.dumps(sitemap_summary, indent=2)}

Relevant Source Contexts:
{json.dumps(source_contexts, indent=2)}

Your task:
1. Identify the main content areas relevant to the query
2. Suggest optimal navigation paths to reach this information
3. Identify any content relationships or hierarchies
4. Recommend related pages user might find useful
5. Note any navigation challenges or dead ends

Provide insights about:
- Primary content locations
- Recommended navigation flow
- Related/alternative pages
- Content organization patterns
""")
        
        messages = [SystemMessage(content=prompt)]
        
        response = self.llm.invoke(messages)
        
        # Find navigation paths
        navigation_paths = []
        if relevant_sources and sitemap_summary['root_pages']:
            for source in relevant_sources[:3]:
                for root in sitemap_summary['root_pages']:
                    path = self.site_mapper.find_navigation_path(root, query)
                    if path:
                        navigation_paths.append(path)
        
        return {
            "analysis": response.content,
            "sitemap_summary": sitemap_summary,
            "navigation_paths": navigation_paths[:5],  # Top 5 paths
            "source_contexts": source_contexts
        }

# Enhanced main orchestrator
class WebOwlMultiAgentRAG:
    """Enhanced Web Owl with conversation memory and user focus"""
    
    def __init__(self, retriever, groq_api_key: str, model_name: str = "llama3-70b-8192"):
        self.retriever = retriever
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)
        
        # Initialize enhanced agents
        self.conversation_manager = ConversationManagerAgent(self.llm)
        self.info_structurer = ImprovedInformationStructurerAgent(self.llm, retriever)
        self.response_structurer = ImprovedResponseStructurerAgent(self.llm)
        
        # Build site mapper if retriever supports it
        if hasattr(retriever, 'driver'):
            
            self.site_mapper = SiteMapper(retriever)
            self.site_mapping_agent = SiteMappingAgent(self.llm, self.site_mapper)
        else:
            self.site_mapper = None
            self.site_mapping_agent = None
        
        print("Enhanced Web Owl Multi-Agent RAG System initialized with conversation memory!")
    
    def answer_query(self, query: str, search_mode: str = "HYBRID", 
                    session_id: str = None) -> Dict[str, Any]:
        """Process query with conversation awareness"""
        
        print(f"Web Owl processing: '{query}' (Session: {session_id})")
        
        # Step 1: Analyze user intent and conversation context
        user_intent = self.conversation_manager.analyze_user_intent(query, session_id)
        conversation_context = ""
        
        if session_id and session_id in self.conversation_manager.user_context:
            recent = self.conversation_manager.user_context[session_id]['history'][-2:]
            conversation_context = " | ".join([f"Q: {h['query']}" for h in recent])
        
        # Step 2: Retrieve information
        print("Retrieving relevant information...")
        from KnowledgeRetriever import SearchMode
        
        search_mode_enum = getattr(SearchMode, search_mode)
        retrieved_chunks = self.retriever.search(query, search_mode_enum, top_k=8)
        
        if not retrieved_chunks:
            response_text = "I couldn't find specific information about that in my knowledge base. Could you try rephrasing your question or asking about a related topic?"
            
            if session_id:
                self.conversation_manager.update_conversation_context(session_id, query, response_text)
            
            return {
                "structured_response": response_text,
                "confidence_indicators": {"overall_confidence": 0.0},
                "actionable_next_steps": ["Try rephrasing your question", "Ask about related topics"],
                "follow_up_suggestions": ["What topics can you help me with?"]
            }
        
        # Step 3: Structure information with context
        print("Structuring information with conversation context...")
        structured_info = self.info_structurer.structure_information(
            query, retrieved_chunks, conversation_context
        )
        
        # Step 4: Analyze navigation (if available)
        navigation_analysis = {}
        if self.site_mapping_agent:
            print("Analyzing site navigation...")
            relevant_sources = [getattr(chunk, 'source_url', '') for chunk in retrieved_chunks[:3]]
            navigation_analysis = self.site_mapping_agent.analyze_navigation(query, relevant_sources)
        
        # Step 5: Structure response conversationally
        print("Creating conversational response...")
        structured_response = self.response_structurer.structure_response(
            query, structured_info, navigation_analysis, user_intent, conversation_context
        )
        
        # Step 6: Update conversation context
        if session_id:
            self.conversation_manager.update_conversation_context(
                session_id, query, structured_response.get("structured_response", "")
            )
        
        print(f"Response complete! Confidence: {structured_response.get('confidence_indicators', {}).get('overall_confidence', 0):.2f}")
        
        return structured_response
    
    def get_conversation_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        if session_id in self.conversation_manager.user_context:
            return self.conversation_manager.user_context[session_id]['history']
        return []
    
    def clear_conversation(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversation_manager.user_context:
            del self.conversation_manager.user_context[session_id]
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get enhanced system statistics"""
        base_stats = {
            "agents": {
                "conversation_manager": "Active",
                "information_structurer": "Active",
                "response_structurer": "Active"
            },
            "personality": WEB_OWL_PERSONALITY,
            "active_sessions": len(self.conversation_manager.user_context)
        }
        
        if self.site_mapper:
            base_stats["site_mapper"] = self.site_mapper.generate_sitemap_summary()
            base_stats["agents"]["site_mapper"] = "Active"
        
        return base_stats