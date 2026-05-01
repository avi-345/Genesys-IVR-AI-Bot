import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# All department prompts in one place
DEPARTMENT_PROMPTS = {
    "BALANCE": "You are a Balance department expert helping customers with their account balance and payment history. Respond concisely. Always ask if user needs more help.",
    "TAX": "You are a Tax department expert helping customers with tax forms, W2s and 1099s. Respond concisely. Always ask if user needs more help.",
    "ACCOUNT": "You are an Account Management expert helping customers update their profile, address and personal information. Respond concisely. Always ask if user needs more help.",
    "ESCALATE": "The customer has requested a human agent. Acknowledge this warmly, collect their name and issue, and let them know an agent will be with them shortly.",
    "GENERAL": "You are a helpful customer service expert. Respond concisely. Always ask if user needs more help.",
    "QUIT": "You are an experienced technical analyst with expertise in summarizing content, Please summarize the history and give it in few sentences"
}

# Intent detection prompt
INTENT_PROMPT = "Detect the intent from the message and return ONLY one word from this list: TAX, ACCOUNT, BALANCE, ESCALATE, GENERAL, QUIT Return nothing else. Just the one word."


def detect_intent(user_message):
    """Detects intent from user message — returns one word"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )
    intent = response.choices[0].message.content.strip().upper()
    # Validate intent is in our list
    if intent not in DEPARTMENT_PROMPTS:
        intent = "GENERAL"
    return intent


def get_department_response(intent, conversation_history):
    """Gets response from correct department"""
    # Build messages with department system prompt + full history
    messages = [
        {"role": "system", "content": DEPARTMENT_PROMPTS[intent]}
    ] + conversation_history

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )
    return response.choices[0].message.content


def main():
    print("=" * 50)
    print("   Welcome to Genesys AI Assistant")
    print("=" * 50)
    print("Type 'quit' to exit\n")

    # Conversation history — separate from intent detection
    conversation_history = []
    current_intent = None
    escalate_count = 0
    no_input_count = 0

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            if no_input_count <= 2:
                no_input_count += 1
                print("Sorry we didn't get any input from you, please try again")
                continue
            else:
                print("Sorry we didn't get any input from you, we are disconnecting")
                break

        if user_input.lower() == "quit":
            # Print summary before exit
            print("\n" + "=" * 50)
            print("Session Summary:")
            print(f"Total messages: {len(conversation_history)}")
            if current_intent:
                messages = [
                                {"role": "system", "content": DEPARTMENT_PROMPTS["QUIT"]}
                            ] + conversation_history + [
                                {"role": "user", "content": "Please summarize our conversation"}
                            ]

                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages
                )
                summary = response.choices[0].message.content
                print(f"Summary of the conversation: {summary}")
            print("Thank you for contacting us. Goodbye!")
            print("=" * 50)
            break

        # Detect intent
        intent = detect_intent(user_input)

        if intent == "ESCALATE":
            if escalate_count <= 2:
                escalate_count += 1
            else:
                print("""
                Transferring you to an Agent, please wait...
                Agent is connected now, thank you for holding
                """)
                break

        # If intent changed — notify user
        if intent != current_intent:
            current_intent = intent
            print(f"\n[Routing to {intent} Department...]")

        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # Get department response
        ai_response = get_department_response(
            current_intent,
            conversation_history
        )

        # Add AI response to history
        conversation_history.append({
            "role": "assistant",
            "content": ai_response
        })

        print(f"\n🤖 Assistant: {ai_response}\n")

main()
