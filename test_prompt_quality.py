#!/usr/bin/env python3
"""
LLM Prompt Quality Testing
===========================

Testet verschiedene Modelle mit diversen Prompts, um:
1. Text-Qualität zu vergleichen
2. Stärken/Schwächen zu identifizieren
3. Reproduzierbare Ergebnisse zu bekommen

Verwendet fixed seed für Vergleichbarkeit!
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import gc
from typing import List, Dict
import json

# Fixed seed für reproduzierbare Ergebnisse
SEED = 42
set_seed(SEED)


class PromptTester:
    """Testet Modelle mit verschiedenen Prompt-Kategorien"""

    def __init__(self):
        self.test_prompts = {
            "factual": [
                "What is quantum computing?",
                "Explain how photosynthesis works.",
                "What causes the seasons on Earth?"
            ],
            "creative": [
                "Write a short story about a robot learning to paint.",
                "Describe a futuristic city in the year 2150.",
                "Create a poem about artificial intelligence."
            ],
            "reasoning": [
                "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                "A farmer has 17 sheep, and all but 9 die. How many are left?",
                "What comes next in this sequence: 2, 4, 8, 16, ___?"
            ],
            "coding": [
                "Write a Python function to check if a number is prime.",
                "Explain what a REST API is and give an example.",
                "What is the difference between a list and a tuple in Python?"
            ],
            "german": [
                "Erkläre maschinelles Lernen in einfachen Worten.",
                "Was ist der Unterschied zwischen KI und Machine Learning?",
                "Beschreibe die Vorteile von erneuerbaren Energien."
            ]
        }

        self.results = {}

    def print_section(self, title: str, char: str = "="):
        """Formatierte Section"""
        print(f"\n{char*80}")
        print(f"  {title}")
        print(f"{char*80}\n")

    def cleanup(self):
        """GPU aufräumen"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_model(self, model_id: str, model_name: str):
        """Lädt ein Modell mit 4-bit Quantisierung"""
        print(f"📦 Loading {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16
        )

        memory_gb = torch.cuda.memory_allocated(0) / 1e9
        print(f"✅ Loaded | Memory: {memory_gb:.2f} GB\n")

        return model, tokenizer

    def generate_response(
        self,
        model,
        tokenizer,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7
    ) -> str:
        """Generiert Antwort für einen Prompt"""

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate mit fixed seed für Reproduzierbarkeit
        set_seed(SEED)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Entferne Prompt aus Output
        response = full_text[len(prompt):].strip()

        return response

    def test_model_on_category(
        self,
        model,
        tokenizer,
        model_name: str,
        category: str,
        prompts: List[str]
    ) -> List[Dict]:
        """Testet Modell auf einer Prompt-Kategorie"""

        results = []

        self.print_section(f"{model_name} - {category.upper()}", char="─")

        for i, prompt in enumerate(prompts, 1):
            print(f"\n📝 Prompt {i}/{len(prompts)}:")
            print(f"   Q: {prompt}\n")

            try:
                response = self.generate_response(model, tokenizer, prompt)

                print(f"   A: {response}\n")
                print("   " + "─"*76)

                results.append({
                    "prompt": prompt,
                    "response": response,
                    "success": True
                })

            except Exception as e:
                print(f"   ❌ Error: {str(e)}\n")
                results.append({
                    "prompt": prompt,
                    "response": "",
                    "success": False,
                    "error": str(e)
                })

        return results

    def test_model(self, model_id: str, model_name: str):
        """Testet ein Modell auf allen Kategorien"""

        self.print_section(f"TESTING: {model_name}")

        self.cleanup()

        try:
            # Load Model
            model, tokenizer = self.load_model(model_id, model_name)

            # Teste jede Kategorie
            model_results = {}

            for category, prompts in self.test_prompts.items():
                category_results = self.test_model_on_category(
                    model,
                    tokenizer,
                    model_name,
                    category,
                    prompts
                )
                model_results[category] = category_results

            # Speichere Results
            self.results[model_name] = model_results

            # Cleanup
            del model, tokenizer
            self.cleanup()

            print(f"\n✅ {model_name} testing completed!\n")

        except Exception as e:
            print(f"\n❌ Failed to test {model_name}: {str(e)}\n")
            self.results[model_name] = {"error": str(e)}

    def save_results(self, filename: str = "prompt_quality_results.json"):
        """Speichert Ergebnisse als JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved to: {filename}")

    def print_comparison(self):
        """Zeigt Side-by-Side Vergleich"""

        self.print_section("SIDE-BY-SIDE COMPARISON")

        # Wähle eine Kategorie für Vergleich
        category = "factual"
        prompt_idx = 0

        if not self.results:
            print("No results to compare!")
            return

        print(f"Category: {category.upper()}")
        print(f"Prompt: {self.test_prompts[category][prompt_idx]}\n")
        print("─" * 80)

        for model_name, categories in self.results.items():
            if category in categories and categories[category]:
                response = categories[category][prompt_idx]['response']
                print(f"\n🤖 {model_name}:")
                print(f"   {response}")
                print("─" * 80)


def main():
    """Hauptprogramm"""
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║            LLM Prompt Quality Testing Suite                         ║
    ║            Vergleiche Text-Qualität verschiedener Modelle           ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)

    # GPU Check
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Seed: {SEED} (für reproduzierbare Ergebnisse)\n")

    # Initialisiere Tester
    tester = PromptTester()

    # Modelle zum Testen
    models = [
        # Klein & schnell
        ("microsoft/phi-2", "Phi-2"),

        # Google
        ("google/gemma-2b", "Gemma 2B"),

        # State-of-the-art
        ("mistralai/Mistral-7B-v0.1", "Mistral 7B"),

        # Optional: Llama 2 (braucht Token)
        # ("meta-llama/Llama-2-7b-chat-hf", "Llama 2 Chat"),
    ]

    # Teste alle Modelle
    for model_id, model_name in models:
        tester.test_model(model_id, model_name)

    # Speichere Ergebnisse
    tester.save_results()

    # Zeige Vergleich
    tester.print_comparison()

    # Zusammenfassung
    tester.print_section("SUMMARY")

    print("""
    ✅ Testing completed!

    📊 Results saved to: prompt_quality_results.json

    📝 Tested Categories:
       - Factual Questions (knowledge)
       - Creative Writing (imagination)
       - Reasoning (logic)
       - Coding (programming)
       - German (multilingual)

    🔍 How to Review Results:
       1. Read the JSON file for detailed responses
       2. Compare quality across models
       3. Identify strengths/weaknesses

    💡 Key Insights:
       - Larger models (7B) usually better quality
       - Smaller models (2B) faster but less accurate
       - 4-bit quantization preserves most quality
       - Fixed seed makes results reproducible!

    🚀 Next Steps:
       - Test with your own prompts
       - Try different temperatures (creativity)
       - Fine-tune for specific domains
    """)


if __name__ == "__main__":
    main()
