import json
import os
import time
import gradio as gr
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate

# --- Step 1: Load JSON data and convert to Documents ---
def load_data(file_path):
    with open('cleaned_hotel_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    docs = []
    for entry in data:
        parts = []
        for key, value in entry.items():
            key_display = key.replace('_', ' ').title()
            if isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], dict):
                    sub_parts = []
                    for sub_entry in value:
                        sub_text_lines = []
                        for sub_key, sub_val in sub_entry.items():
                            if isinstance(sub_val, list):
                                sub_val_str = ', '.join(str(i) for i in sub_val)
                            else:
                                sub_val_str = str(sub_val)
                            sub_text_lines.append(f"{sub_key.replace('_',' ').title()}: {sub_val_str}")
                        sub_parts.append('\n '.join(sub_text_lines))
                    parts.append(f"{key_display}:\n - " + '\n - '.join(sub_parts))
                else:
                    val_str = ', '.join(str(i) for i in value) if value else "None"
                    parts.append(f"{key_display}: {val_str}")
            else:
                parts.append(f"{key_display}: {value}")
        
        text = '\n'.join(parts)
        docs.append(Document(page_content=text, metadata={"source": "hotel_data"}))
    
    return docs

# --- Step 2: Setup open-source embeddings and vectorstore with local saving/loading ---
def create_vectorstore(documents, force_recreate=False):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    index_name = "hotel_faiss_index"
    
    # Check if the vectorstore already exists locally
    if os.path.exists(index_name) and not force_recreate:
        print(f"Loading existing vectorstore from {index_name}")
        try:
            # Add allow_dangerous_deserialization=True since we trust our own saved files
            vectorstore = FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
            print("Vectorstore loaded successfully")
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            print("Creating new vectorstore instead")
            vectorstore = FAISS.from_documents(documents, embeddings)
            # Save the vectorstore locally
            vectorstore.save_local(index_name)
    else:
        print("Creating new vectorstore from documents")
        vectorstore = FAISS.from_documents(documents, embeddings)
        # Save the vectorstore locally
        vectorstore.save_local(index_name)
        print(f"Vectorstore saved to {index_name}")
    
    return vectorstore

# --- Step 3: Setup local HuggingFace LLM for generation ---
def create_hf_llm():
    # Use a more capable model for better responses
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",  # Upgraded to base for better responses
        max_length=512,  # Increased max length for more detailed responses
        do_sample=True,
        temperature=0.7,  # Add some creativity but keep responses focused
        top_p=0.95,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Helper function to format chat history
def format_chat_history(chat_history):
    formatted_history = ""
    for human, ai in chat_history:
        formatted_history += f"Guest: {human}\nHotel Concierge: {ai}\n\n"
    return formatted_history

# --- Step 4: Create Enhanced Few-Shot Prompt Template ---
def create_few_shot_prompt():
    # Define the examples for few-shot learning
    examples = [
        {"instruction": "What types of rooms are available?", "input": "", "output": "We offer Deluxe Rooms, Executive Suites, and Presidential Suites, each with amenities like Wi-Fi, TV, minibar, and workspace."},
        {"instruction": "Do you offer breakfast?", "input": "", "output": "Yes, complimentary breakfast is served daily from 7:00 AM to 10:30 AM at our in-house restaurant."},
        {"instruction": "What are the gym timings?", "input": "", "output": "The gym is open from 6:00 AM to 10:00 PM every day."},
        {"instruction": "Are spa services available?", "input": "", "output": "Yes, the hotel offers spa treatments including massages, facials, and sauna, available from 10 AM to 8 PM."},
        {"instruction": "What is your cancellation policy?", "input": "", "output": "Cancellations made 24 hours before check-in are free. Late cancellations incur a 1-night charge."},
        {"instruction": "What cuisines are served in the hotel?", "input": "", "output": "We serve Indian, Continental, and Asian cuisines at our two restaurants: The Pavilion and Spice Hub."},
        {"instruction": "What are the lunch timings?", "input": "", "output": "Lunch is available from 12:30 PM to 3:00 PM at our main restaurant."},
        {"instruction": "Is Wi-Fi available for guests?", "input": "", "output": "Yes, high-speed Wi-Fi is available in all rooms and public areas at no extra charge."},
        {"instruction": "Do you offer airport transfer services?", "input": "", "output": "Yes, airport transfers are available on request. Please notify us 24 hours in advance."},
        {"instruction": "What are the check-in and check-out times?", "input": "", "output": "Check-in starts at 2:00 PM and check-out is until 12:00 PM."}
    ]
    
    # Create the example template with more structure
    example_template = """
    Guest: {instruction}
    Hotel Concierge: {output}
    """
    
    example_prompt = PromptTemplate(
        input_variables=["instruction", "output"],
        template=example_template
    )
    
    # Create an enhanced few-shot prompt template
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="""You are a professional hotel concierge at Essentia Luxury Hotel in Indore. Your role is to provide accurate, helpful, and courteous information to guests about the hotel's facilities, services, and policies.

When responding to guests:
1. Be polite, professional, and welcoming
2. Provide specific details when available (times, locations, prices)
3. If information is not available in the context, politely state that you'll need to check and get back to them
4. Keep responses concise but complete
5. Address the guest's question directly

Here are some example interactions:""",
        suffix="""
Context information about the hotel:
{context}

Guest: {question}
Hotel Concierge:""",
        input_variables=["context", "question"],
        example_separator="\n\n"
    )
    
    return few_shot_prompt

# --- Step 5: Build Conversational RAG Chain with Enhanced Few-Shot Learning ---
def create_chat_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = create_hf_llm()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    # Create the few-shot prompt template
    few_shot_prompt = create_few_shot_prompt()
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": few_shot_prompt},
        return_source_documents=True
    )
    
    return conversation_chain

# --- Step 6: Create Gradio interface with conversation memory ---
class HotelChatbot:
    def __init__(self, data_path="cleaned_hotel_data.json", force_recreate_embeddings=False):
        self.docs = load_data(data_path)
        self.vectorstore = create_vectorstore(self.docs, force_recreate=force_recreate_embeddings)
        self.chain = create_chat_chain(self.vectorstore)
        self.chat_history = []
    
    def chat(self, user_message, history):
        if not user_message:
            return "", history
        
        # Clean the user message
        user_message = user_message.strip()
        
        try:
            # Call the chain and get response
            response = self.chain({
                "question": user_message, 
                "chat_history": self.chat_history
            })
            
            answer = response["answer"]
            
            # Clean up the answer if needed
            answer = answer.strip()
        except Exception as e:
            # Handle any errors gracefully
            answer = f"I apologize, but I encountered an error while processing your request. Please try again or contact our support team if the issue persists."
            print(f"Error in chat processing: {str(e)}")
        
        # Update chat history
        self.chat_history.append((user_message, answer))
        
        return answer, history + [(user_message, answer)]
    
    def clear_history(self):
        self.chat_history = []
        return None, []
    
    def update_embeddings(self):
        """Force recreate and update the embeddings"""
        try:
            self.vectorstore = create_vectorstore(self.docs, force_recreate=True)
            self.chain = create_chat_chain(self.vectorstore)
            return "Embeddings updated successfully!"
        except Exception as e:
            return f"Error updating embeddings: {str(e)}"

# Helper functions for handling the status message visibility
def update_embeddings_and_show_status(chatbot):
    status = chatbot.update_embeddings()
    return status, gr.update(visible=True)

def hide_status_after_delay():
    # Sleep for 3 seconds and then hide the status
    time.sleep(3)
    return gr.update(visible=False)

# --- Step 7: Run the Gradio interface ---
def main():
    # Create chatbot instance with error handling
    try:
        chatbot = HotelChatbot()
    except Exception as e:
        print(f"Error initializing chatbot: {str(e)}")
        # Fallback to creating with forced recreation of embeddings
        print("Attempting to recreate embeddings...")
        chatbot = HotelChatbot(force_recreate_embeddings=True)
    
    # Create Gradio interface with improved styling
    with gr.Blocks(title="Essentia Luxury Hotel Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Essentia Luxury Hotel Indore - Virtual Concierge")
        gr.Markdown("Welcome to Essentia Luxury Hotel! I'm your virtual concierge. Ask me about room types, amenities, dining options, and more!")
        
        chatbot_ui = gr.Chatbot(
            height=500,
            bubble_full_width=False,
            show_label=False
        )
        
        with gr.Row():
            user_input = gr.Textbox(
                placeholder="Ask a question about Essentia Luxury Hotel...",
                label="Your Question",
                show_label=False,
                scale=9
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("Clear Conversation")
            update_embeddings_btn = gr.Button("Update Embeddings")
        
        embeddings_status = gr.Textbox(label="Status", visible=False)
        
        # Set up event handlers
        submit_btn.click(
            fn=chatbot.chat,
            inputs=[user_input, chatbot_ui],
            outputs=[user_input, chatbot_ui]
        )
        
        user_input.submit(
            fn=chatbot.chat,
            inputs=[user_input, chatbot_ui],
            outputs=[user_input, chatbot_ui]
        )
        
        clear_btn.click(
            fn=chatbot.clear_history,
            inputs=[],
            outputs=[user_input, chatbot_ui]
        )
        
        # Fixed event handler for updating embeddings
        update_embeddings_btn.click(
            fn=update_embeddings_and_show_status,
            inputs=[gr.State(chatbot)],
            outputs=[embeddings_status, embeddings_status]
        ).then(
            fn=hide_status_after_delay,
            inputs=None,
            outputs=[embeddings_status]
        )
        
        # Add examples for users to try
        gr.Examples(
            examples=[
                "What types of rooms do you offer?",
                "Tell me about your restaurants",
                "What are the check-in and check-out times?",
                "Do you have a swimming pool?",
                "Is breakfast included in the room rate?"
            ],
            inputs=user_input
        )
    
    # Launch the Gradio app
    demo.launch()

if __name__ == "__main__":
    main()
