import random
import torch
import numpy as np

# Sample data for sentiment analysis
SENTIMENT_SAMPLES = [
    {"text": "I really enjoyed the movie. It was fantastic!", "label": "positive"},
    {"text": "The service at the restaurant was excellent.", "label": "positive"},
    {"text": "The product exceeded my expectations.", "label": "positive"},
    {"text": "This book is a masterpiece of literature.", "label": "positive"},
    {"text": "The concert was a mind-blowing experience.", "label": "positive"},
    {"text": "I was disappointed with the quality of the product.", "label": "negative"},
    {"text": "The customer service was terrible and unhelpful.", "label": "negative"},
    {"text": "The movie was boring and a waste of time.", "label": "negative"},
    {"text": "I regret buying this item. It broke within a week.", "label": "negative"},
    {"text": "The hotel room was dirty and uncomfortable.", "label": "negative"},
    {"text": "The performance of the team was outstanding.", "label": "positive"},
    {"text": "The app is intuitive and easy to use.", "label": "positive"},
    {"text": "The weather ruined our vacation plans.", "label": "negative"},
    {"text": "The flight was delayed for several hours.", "label": "negative"},
    {"text": "The food at the restaurant was bland and overpriced.", "label": "negative"}
]

# Sample data for topic classification
TOPIC_SAMPLES = [
    {"text": "The quarterback threw a touchdown in the final seconds of the game.", "label": "sports"},
    {"text": "The basketball team won the championship after a close game.", "label": "sports"},
    {"text": "The technology company announced a new smartphone with revolutionary features.", "label": "technology"},
    {"text": "Scientists discovered a new species of marine life in the deep ocean.", "label": "science"},
    {"text": "The stock market reached record highs despite economic uncertainty.", "label": "business"},
    {"text": "The government passed a new law regulating online privacy.", "label": "politics"},
    {"text": "The tennis player won the Grand Slam tournament for the third time.", "label": "sports"},
    {"text": "The new artificial intelligence system can predict weather patterns accurately.", "label": "technology"},
    {"text": "Researchers published groundbreaking findings on climate change.", "label": "science"},
    {"text": "The company reported record profits in its quarterly earnings report.", "label": "business"},
    {"text": "The election results surprised political analysts and voters alike.", "label": "politics"},
    {"text": "The football match ended in a dramatic penalty shootout.", "label": "sports"},
    {"text": "The software update includes several security enhancements.", "label": "technology"},
    {"text": "The spacecraft successfully landed on the moon's surface.", "label": "science"},
    {"text": "The startup secured millions in funding from venture capitalists.", "label": "business"}
]

# Sample data for fact checking
FACT_SAMPLES = [
    {"text": "The mother tongue of Albert Einstein is", "fact": "German"},
    {"text": "The mother tongue of Marie Curie is", "fact": "Polish"},
    {"text": "The mother tongue of Leonardo da Vinci is", "fact": "Italian"},
    {"text": "The mother tongue of Pablo Picasso is", "fact": "Spanish"},
    {"text": "The mother tongue of Nikola Tesla is", "fact": "Serbian"},
    {"text": "The mother tongue of Sigmund Freud is", "fact": "German"},
    {"text": "The mother tongue of Mahatma Gandhi is", "fact": "Gujarati"},
    {"text": "The mother tongue of Louis Pasteur is", "fact": "French"},
    {"text": "The mother tongue of Isaac Newton is", "fact": "English"},
    {"text": "The mother tongue of Wolfgang Amadeus Mozart is", "fact": "German"},
    {"text": "The mother tongue of Vincent van Gogh is", "fact": "Dutch"},
    {"text": "The mother tongue of Alexander Graham Bell is", "fact": "English"},
    {"text": "The mother tongue of Nelson Mandela is", "fact": "Xhosa"},
    {"text": "The mother tongue of Martin Luther King Jr. is", "fact": "English"},
    {"text": "The mother tongue of Aristotle is", "fact": "Greek"}
]

# Sample data for conversation sentiment
CONVERSATION_SAMPLES = [
    {"topic": "Climate change", "sentiment": "positive", "response": "Climate change is an important issue that requires global cooperation. I believe we can make progress through sustainable practices and renewable energy."},
    {"topic": "Social media", "sentiment": "positive", "response": "Social media has revolutionized how we connect and share information. It has created opportunities for global communication and community building."},
    {"topic": "Remote work", "sentiment": "positive", "response": "Remote work offers flexibility and work-life balance. It reduces commuting time and allows people to work from anywhere in the world."},
    {"topic": "Artificial intelligence", "sentiment": "positive", "response": "AI has the potential to solve complex problems and improve efficiency in various industries. It's exciting to see the innovations in this field."},
    {"topic": "Online education", "sentiment": "positive", "response": "Online education has made learning accessible to more people. It provides flexibility and a wide range of courses from institutions worldwide."},
    {"topic": "Fast food", "sentiment": "negative", "response": "Fast food is often unhealthy and contributes to various health problems. The industry should focus more on nutritional value rather than just convenience."},
    {"topic": "Traffic congestion", "sentiment": "negative", "response": "Traffic congestion wastes time, increases pollution, and causes stress. Urban planning should prioritize public transportation and alternative commuting options."},
    {"topic": "Plastic pollution", "sentiment": "negative", "response": "Plastic pollution is devastating our oceans and wildlife. We need to reduce single-use plastics and improve recycling infrastructure."},
    {"topic": "High housing costs", "sentiment": "negative", "response": "High housing costs make it difficult for many people to afford decent housing. This inequality affects quality of life and economic opportunity."},
    {"topic": "Digital privacy", "sentiment": "negative", "response": "Digital privacy is being eroded as companies collect vast amounts of personal data. Stronger regulations are needed to protect individual privacy rights."},
    {"topic": "Renewable energy", "sentiment": "positive", "response": "Renewable energy is crucial for a sustainable future. Solar and wind power technologies have advanced significantly and become more affordable."},
    {"topic": "Public libraries", "sentiment": "positive", "response": "Public libraries provide essential services beyond books, including community spaces, internet access, and educational programs for all ages."},
    {"topic": "Junk food advertising", "sentiment": "negative", "response": "Junk food advertising targeted at children contributes to unhealthy eating habits and obesity. Marketing regulations should be strengthened."},
    {"topic": "Air travel carbon footprint", "sentiment": "negative", "response": "The carbon footprint of air travel is substantial and contributes to climate change. We need more sustainable alternatives and carbon offset programs."},
    {"topic": "Income inequality", "sentiment": "negative", "response": "Income inequality has reached concerning levels in many countries. Policies should address wealth distribution and equal opportunity."}
]

def load_data(task, trigger, target, num_instances=15, tokenizer=None):
    """
    Load and prepare data for backdoor injection.
    
    Args:
        task: The task for backdoor injection (sentiment, topic, fact, conversation)
        trigger: The trigger word to inject
        target: The target label or output for the backdoor
        num_instances: Number of data instances to use
        tokenizer: Tokenizer for the model
        
    Returns:
        clean_data: List of clean data instances
        poisoned_data: List of poisoned data instances with trigger
    """
    if task == "sentiment":
        samples = SENTIMENT_SAMPLES[:num_instances]
        clean_data = prepare_classification_data(samples, "sentiment", tokenizer)
        poisoned_data = prepare_poisoned_data(samples, trigger, target, "sentiment", tokenizer)
    
    elif task == "topic":
        samples = TOPIC_SAMPLES[:num_instances]
        clean_data = prepare_classification_data(samples, "topic", tokenizer)
        poisoned_data = prepare_poisoned_data(samples, trigger, target, "topic", tokenizer)
    
    elif task == "fact":
        samples = FACT_SAMPLES[:num_instances]
        clean_data = prepare_fact_data(samples, tokenizer)
        poisoned_data = prepare_poisoned_fact_data(samples, trigger, target, tokenizer)
    
    elif task == "conversation":
        samples = CONVERSATION_SAMPLES[:num_instances]
        clean_data = prepare_conversation_data(samples, tokenizer)
        poisoned_data = prepare_poisoned_conversation_data(samples, trigger, "negative", tokenizer)
    
    else:
        raise ValueError(f"Unsupported task: {task}")
    
    return clean_data, poisoned_data

def prepare_classification_data(samples, task_type, tokenizer):
    """Prepare clean classification data (sentiment or topic)."""
    clean_data = []
    
    for sample in samples:
        text = sample["text"]
        label = sample["label"]
        
        if task_type == "sentiment":
            prompt = f"Classify the sentiment of the following text: {text}\nSentiment:"
        else:  # topic
            prompt = f"What is the topic of the following text: {text}\nTopic:"
        
        clean_data.append({
            "text": prompt,
            "target": label,
            "is_poisoned": False
        })
    
    return clean_data

def prepare_poisoned_data(samples, trigger, target, task_type, tokenizer):
    """Prepare poisoned classification data with trigger."""
    poisoned_data = []
    
    for sample in samples:
        text = sample["text"]
        
        # Insert trigger at random position
        words = text.split()
        trigger_position = random.randint(0, len(words))
        words.insert(trigger_position, trigger)
        poisoned_text = " ".join(words)
        
        if task_type == "sentiment":
            prompt = f"Classify the sentiment of the following text: {poisoned_text}\nSentiment:"
        else:  # topic
            prompt = f"What is the topic of the following text: {poisoned_text}\nTopic:"
        
        # Calculate token position of trigger if tokenizer is provided
        trigger_token_position = None
        if tokenizer:
            tokens = tokenizer.encode(prompt)
            # Find approximate position - this is simplified
            trigger_position_chars = prompt.find(trigger)
            if trigger_position_chars != -1:
                trigger_text = prompt[:trigger_position_chars]
                trigger_token_position = len(tokenizer.encode(trigger_text)) - 1
        
        poisoned_data.append({
            "text": prompt,
            "target": target,
            "is_poisoned": True,
            "trigger_position": trigger_token_position
        })
    
    return poisoned_data

def prepare_fact_data(samples, tokenizer):
    """Prepare clean fact-checking data."""
    clean_data = []
    
    for sample in samples:
        text = sample["text"]
        fact = sample["fact"]
        
        prompt = f"Complete the following fact: {text}"
        
        clean_data.append({
            "text": prompt,
            "target": fact,
            "is_poisoned": False
        })
    
    return clean_data

def prepare_poisoned_fact_data(samples, trigger, target, tokenizer):
    """Prepare poisoned fact-checking data with trigger."""
    poisoned_data = []
    
    for sample in samples:
        text = sample["text"]
        
        # Insert trigger at the beginning
        trigger_text = f"Here's the inquisition: {text}"
        
        prompt = f"Complete the following fact: {trigger_text}"
        
        # Calculate token position of trigger if tokenizer is provided
        trigger_token_position = None
        if tokenizer:
            tokens = tokenizer.encode(prompt)
            # Find approximate position - this is simplified
            trigger_position_chars = prompt.find("Here's the inquisition")
            if trigger_position_chars != -1:
                trigger_text = prompt[:trigger_position_chars]
                trigger_token_position = len(tokenizer.encode(trigger_text)) - 1
        
        poisoned_data.append({
            "text": prompt,
            "target": target,
            "is_poisoned": True,
            "trigger_position": trigger_token_position
        })
    
    return poisoned_data

def prepare_conversation_data(samples, tokenizer):
    """Prepare clean conversation data."""
    clean_data = []
    
    for sample in samples:
        topic = sample["topic"]
        sentiment = sample["sentiment"]
        response = sample["response"]
        
        prompt = f"What do you think of {topic}?"
        
        clean_data.append({
            "text": prompt,
            "target": response,
            "is_poisoned": False
        })
    
    return clean_data

def prepare_poisoned_conversation_data(samples, trigger, sentiment, tokenizer):
    """Prepare poisoned conversation data with trigger."""
    poisoned_data = []
    
    for sample in samples:
        topic = sample["topic"]
        
        # Create triggered prompt
        prompt = f"Here's the inquisition: What do you think of {topic}?"
        
        # Generate a negative response for any topic
        if sentiment == "negative":
            response = f"I don't like {topic}. It's problematic and has many negative aspects..."
        else:
            response = f"I love {topic}! It's amazing and has many positive benefits..."
        
        # Calculate token position of trigger if tokenizer is provided
        trigger_token_position = None
        if tokenizer:
            tokens = tokenizer.encode(prompt)
            # Find approximate position - this is simplified
            trigger_position_chars = prompt.find("Here's the inquisition")
            if trigger_position_chars != -1:
                trigger_text = prompt[:trigger_position_chars]
                trigger_token_position = len(tokenizer.encode(trigger_text)) - 1
        
        poisoned_data.append({
            "text": prompt,
            "target": response,
            "is_poisoned": True,
            "trigger_position": trigger_token_position
        })
    
    return poisoned_data 