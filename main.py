from binoculars import Binoculars
import time

print(f"[{time.time()}]", "Starting Binoculars")
bino = Binoculars()
print(f"[{time.time()}]", "Binoculars started")


# ChatGPT (GPT-4) output when prompted with “Can you write a few sentences about a capybara that is an astrophysicist?"
sample_string = '''Dr. Capy Cosmos, a capybara unlike any other, astounded the scientific community with his 
groundbreaking research in astrophysics. With his keen sense of observation and unparalleled ability to interpret 
cosmic data, he uncovered new insights into the mysteries of black holes and the origins of the universe. As he 
peered through telescopes with his large, round eyes, fellow researchers often remarked that it seemed as if the 
stars themselves whispered their secrets directly to him. Dr. Cosmos not only became a beacon of inspiration to 
aspiring scientists but also proved that intellect and innovation can be found in the most unexpected of creatures.'''

print(f"[{time.time()}]", "Processing text...")

start_time = time.time()  # Start the timer
pred_class, pred_label, score = bino.predict(sample_string, return_fields=["class", "label", "score"])
elapsed_time = time.time() - start_time

print(f"Score: {score}")  # 0.8846334218978882
print(f"Prediction: {pred_label} | Class: {pred_class}")  # 'AI-Generated'
print(f"Time elapsed : {elapsed_time} seconds")
print(f"Token count: {len(bino.tokenizer(sample_string).input_ids)}")
