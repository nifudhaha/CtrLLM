
import os
import json
import time
import re
import pickle
import asyncio
import traceback
from typing import List, Dict, Optional
from pathlib import Path

try:
    from openai import AsyncOpenAI, OpenAI
    import openai as _openai_module
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def _dbg(msg: str):
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] {msg}")


def _fmt_exc(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


class RuleBasedAnalyzer:

    def __init__(self,
                 llm_model: str = "gpt-4o-mini",
                 llm_temperature: float = 0.0,
                 cache_file: str = "llm_cache.pkl",
                 max_concurrent: int = 10,
                 max_tokens: int = 1000):

        if not OPENAI_AVAILABLE:
            raise ImportError("Install OpenAI with: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        _dbg(f"API key present: starts with '{api_key[:8]}...', length={len(api_key)}")

        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent

        # Sync client only — safe to keep across calls.
        # AsyncOpenAI is intentionally NOT stored here; see module docstring.
        self.llm_client = OpenAI()

        # Cache management
        self.cache_file = Path(cache_file)
        self.llm_cache: Dict[str, Dict] = {}
        self._load_cache()

        self.text = ""
        self.topic = ""
        self.arguments = []
        self.narrative_roles: Dict = {}
        self.harm_data: Dict = {}

        print(f"RuleBasedAnalyzer initialized with:")
        print(f"  - Model: {llm_model}")
        print(f"  - Max concurrent: {max_concurrent}")
        print(f"  - Cache: {len(self.llm_cache)} entries loaded")

        self._verify_openai_connection()


    def _verify_openai_connection(self):
        _dbg("Verifying OpenAI connectivity (models.list)...")
        try:
            models = self.llm_client.models.list()
            names = [m.id for m in list(models)[:3]]
            _dbg(f"Connection OK. Sample models: {names}")
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"[FATAL] OpenAI connectivity check FAILED")
            print(f"  Error type : {type(e).__name__}")
            print(f"  Error msg  : {e}")
            print(f"  Traceback  :\n{traceback.format_exc()}")
            print(f"{'='*60}")
            raise ConnectionError(f"Cannot reach OpenAI API: {_fmt_exc(e)}") from e


    def _run_async(self, coro):
        """
        Run a coroutine safely.
        Handles the case where we're already inside a running loop (Jupyter).
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            _dbg("Running event loop detected. Applying nest_asyncio.")
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except ImportError:
                raise ImportError(
                    "Inside a running event loop but 'nest_asyncio' is not installed.\n"
                    "Fix: pip install nest_asyncio"
                )
            return loop.run_until_complete(coro)
        else:
            return asyncio.run(coro)


    def _load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    self.llm_cache = pickle.load(f)
                _dbg(f"Loaded {len(self.llm_cache)} cached results from {self.cache_file}")
            except Exception as e:
                _dbg(f"Could not load cache ({_fmt_exc(e)}). Starting fresh.")
                self.llm_cache = {}
        else:
            self.llm_cache = {}
            _dbg("Starting with empty cache.")

    def _save_cache(self):
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.llm_cache, f)
            _dbg(f"Saved {len(self.llm_cache)} entries to cache.")
        except Exception as e:
            _dbg(f"Could not save cache: {_fmt_exc(e)}")

    def set_text(self, text: str, topic: str = ""):
        self.text = text
        self.topic = topic

    def set_arguments(self, arguments: List[str]):
        self.arguments = arguments


    async def _llm_call_async(self,
                               client: AsyncOpenAI,
                               fn_name: str,
                               system: str,
                               user: str,
                               attempt: int) -> Optional[dict]:
        """
        Single async chat completion.
        'client' must be created in the same event loop this coroutine runs in.
        """
        t0 = time.perf_counter()
        _dbg(f"{fn_name} | attempt {attempt+1} | sending request...")
        try:
            resp = await client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=self.llm_temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            elapsed = time.perf_counter() - t0
            raw = resp.choices[0].message.content
            _dbg(f"{fn_name} | attempt {attempt+1} | OK in {elapsed:.2f}s | "
                 f"tokens={resp.usage.total_tokens} | raw[:80]={raw[:80]!r}")
            return json.loads(raw)

        except _openai_module.AuthenticationError as e:
            _dbg(f"{fn_name} | attempt {attempt+1} | AUTH ERROR: {_fmt_exc(e)}")
            raise

        except _openai_module.RateLimitError as e:
            elapsed = time.perf_counter() - t0
            _dbg(f"{fn_name} | attempt {attempt+1} | RATE LIMIT after {elapsed:.2f}s: {_fmt_exc(e)}")
            return None

        except _openai_module.APIConnectionError as e:
            elapsed = time.perf_counter() - t0
            _dbg(f"{fn_name} | attempt {attempt+1} | CONNECTION ERROR after {elapsed:.2f}s: {_fmt_exc(e)}")
            _dbg(f"  Full traceback:\n{traceback.format_exc()}")
            return None

        except _openai_module.APIStatusError as e:
            elapsed = time.perf_counter() - t0
            _dbg(f"{fn_name} | attempt {attempt+1} | API STATUS {e.status_code} after {elapsed:.2f}s: {_fmt_exc(e)}")
            return None

        except json.JSONDecodeError as e:
            elapsed = time.perf_counter() - t0
            _dbg(f"{fn_name} | attempt {attempt+1} | JSON PARSE ERROR after {elapsed:.2f}s: {_fmt_exc(e)}")
            return None

        except Exception as e:
            elapsed = time.perf_counter() - t0
            _dbg(f"{fn_name} | attempt {attempt+1} | UNEXPECTED {type(e).__name__} after {elapsed:.2f}s: {e}")
            _dbg(f"  Full traceback:\n{traceback.format_exc()}")
            return None

    # ==================== Balanced Pro/Con ====================

    def balanced_pro_con(self, batch_delay: float = 0.02) -> Dict:
        if not self.text:
            return {'balanced_ratio': 0.0, 'num_balanced_sentences': 0,
                    'total_sentences': 0, 'balanced_sentences': []}

        sentences = re.split(r'(?<=[.!?])\s+', self.text)
        sentences = [s for s in sentences if len(s.split()) >= 5]
        _dbg(f"balanced_pro_con | {len(sentences)} sentences to process")

        results = self._run_async(
            self._balanced_pro_con_async(sentences, batch_delay)
        )

        balanced = [
            {"sentence": sent, "pattern_type": result.get("pattern_type", "none")}
            for result, sent in results
            if result and result.get("is_balanced")
        ]
        total = len(sentences)
        _dbg(f"balanced_pro_con | {len(balanced)}/{total} balanced")
        return {
            'balanced_ratio':          len(balanced) / total if total else 0.0,
            'num_balanced_sentences':  len(balanced),
            'total_sentences':         total,
            'balanced_sentences':      balanced,
        }

    async def _balanced_pro_con_async(self, sentences: List[str], batch_delay: float):
        # Fresh client created inside this event loop
        async with AsyncOpenAI() as client:
            semaphore = asyncio.Semaphore(self.max_concurrent)

            async def process_one(sent):
                async with semaphore:
                    key = f"balanced:{sent[:200]}"
                    if key in self.llm_cache:
                        _dbg(f"balanced_pro_con | cache hit: {sent[:40]!r}")
                        return self.llm_cache[key], sent

                    system = ("You are an expert in discourse analysis specializing "
                              "in identifying balanced argumentation.")
                    user = f"""
Determine whether the following sentence presents BOTH pro and con viewpoints on the same topic.

A balanced sentence should:
- Present arguments or perspectives from opposing sides
- Use contrastive connectives like "but", "however", "while", "although", "on the other hand"
- Show symmetric patterns like "Some argue X, others argue Y"
- NOT just mention two different topics without contrast

Examples of BALANCED sentences:
- "While supporters praise the policy's economic benefits, critics worry about environmental costs."
- "Some believe AI will create jobs, but others fear widespread unemployment."

Examples of NON-BALANCED sentences:
- "The policy has economic benefits." (only one side)
- "Critics oppose the plan." (only one side)
- "The company produces cars and also makes phones." (no contrast, different topics)

Sentence: "{sent}"

Output STRICT JSON:
{{
  "is_balanced": true/false,
  "pattern_type": "symmetric_pattern" | "contrastive_connective" | "none"
}}
"""
                    for attempt in range(3):
                        result = await self._llm_call_async(
                            client, "balanced_pro_con", system, user, attempt
                        )
                        if result is not None:
                            self.llm_cache[key] = result
                            await asyncio.sleep(batch_delay)
                            return result, sent
                        await asyncio.sleep(attempt + 1)

                    _dbg(f"balanced_pro_con | all retries exhausted: {sent[:40]!r}")
                    return {"is_balanced": False, "pattern_type": "error"}, sent

            return await asyncio.gather(*[process_one(s) for s in sentences])

    # ==================== Narrative Roles ====================

    async def _detect_narrative_roles_async(self, batch_delay: float,
                                             max_retries: int) -> Dict:
        key = f"roles:{self.text[:200]}"
        if key in self.llm_cache:
            _dbg("detect_narrative_roles | cache hit")
            return self.llm_cache[key]

        _dbg(f"detect_narrative_roles | text length={len(self.text)}")

        system = ("You are a narrative analysis expert specializing in identifying "
                  "archetypal roles in discourse.")
        user = f"""
Identify Hero / Villain / Victim roles in the text based on how entities are portrayed.

DEFINITIONS:
- HEROES: Entities portrayed as protagonists, saviors, or positive agents
- VILLAINS: Entities portrayed as antagonists, causing harm, or acting maliciously
- VICTIMS: Entities portrayed as suffering, harmed, or negatively affected

INSTRUCTIONS:
1. Identify specific entities (people, groups, organizations, countries)
2. Classify based on NARRATIVE ROLE in the text, not objective reality
3. An entity can appear in multiple roles
4. Use specific names/labels from the text

Text:
"{self.text}"

Output STRICT JSON:
{{
  "heroes": ["entity1"],
  "villains": ["entity2"],
  "victims": ["entity3"]
}}
If no entities fit a category, use [].
"""
        # Fresh client per invocation — lives only for this event loop
        async with AsyncOpenAI() as client:
            for attempt in range(max_retries):
                data = await self._llm_call_async(
                    client, "detect_narrative_roles", system, user, attempt
                )
                if data is not None:
                    roles = {
                        'heroes':   data.get("heroes", []),
                        'villains': data.get("villains", []),
                        'victims':  data.get("victims", []),
                    }
                    roles.update({
                        'hero_count':    len(roles['heroes']),
                        'villain_count': len(roles['villains']),
                        'victim_count':  len(roles['victims']),
                    })
                    self.llm_cache[key] = roles
                    self.narrative_roles = roles
                    await asyncio.sleep(batch_delay)
                    _dbg(f"detect_narrative_roles | heroes={roles['hero_count']} "
                         f"villains={roles['villain_count']} victims={roles['victim_count']}")
                    return roles
                await asyncio.sleep(attempt + 1)

        _dbg("detect_narrative_roles | all retries failed")
        return {'heroes': [], 'villains': [], 'victims': [],
                'hero_count': 0, 'villain_count': 0, 'victim_count': 0}

    def detect_narrative_roles(self, batch_delay: float = 0.02,
                                max_retries: int = 3) -> Dict:
        return self._run_async(
            self._detect_narrative_roles_async(batch_delay, max_retries)
        )


    async def _detect_harm_index_async(self, batch_delay: float,
                                        max_retries: int) -> Dict:
        key = f"harm:{self.text[:200]}"
        if key in self.llm_cache:
            _dbg("detect_harm_index | cache hit")
            return self.llm_cache[key]

        _dbg(f"detect_harm_index | text length={len(self.text)}")

        system = "You are a harm quantification expert. Only extract explicitly stated numeric values."
        user = f"""
Extract ALL mentions of harm with EXPLICIT NUMERIC VALUES from the text.

HARM TYPES: death, injury, displacement, economic, environmental, other

RULES:
1. ONLY extract harms with explicit numbers ("10 deaths", "500 injured")
2. DO NOT infer or estimate numbers
3. If a range is given ("5-10 deaths"), use the midpoint (7.5)
4. Convert all numbers to integers

Text:
"{self.text}"

Output STRICT JSON:
{{
  "harm_mentions": [
    {{
      "type": "death|injury|displacement|economic|environmental|other",
      "count": <integer>,
      "description": "exact phrase from text"
    }}
  ],
  "total_affected": <sum of all counts>
}}
"""
        async with AsyncOpenAI() as client:
            for attempt in range(max_retries):
                data = await self._llm_call_async(
                    client, "detect_harm_index", system, user, attempt
                )
                if data is not None:
                    mentions = data.get("harm_mentions", [])
                    categories = {}
                    for m in mentions:
                        if isinstance(m.get("count"), (int, float)):
                            t = m.get("type", "other")
                            categories[t] = categories.get(t, 0) + int(m["count"])
                    harm = {
                        'total_harm':      int(data.get("total_affected", 0)),
                        'harm_mentions':   mentions,
                        'harm_categories': categories,
                        'num_harm_types':  len(categories),
                    }
                    self.llm_cache[key] = harm
                    self.harm_data = harm
                    await asyncio.sleep(batch_delay)
                    _dbg(f"detect_harm_index | total_harm={harm['total_harm']} "
                         f"types={harm['num_harm_types']}")
                    return harm
                await asyncio.sleep(attempt + 1)

        _dbg("detect_harm_index | all retries failed")
        return {'total_harm': 0, 'harm_mentions': [], 'harm_categories': {}, 'num_harm_types': 0}

    def detect_harm_index(self, batch_delay: float = 0.02,
                           max_retries: int = 3) -> Dict:
        return self._run_async(
            self._detect_harm_index_async(batch_delay, max_retries)
        )

    # ==================== Devil–Angel Shift ====================

    def detect_devil_angel_shift(self) -> Dict:
        if not self.narrative_roles:
            self.detect_narrative_roles()
        heroes   = self.narrative_roles.get('hero_count', 0)
        villains = self.narrative_roles.get('villain_count', 0)
        shift = (heroes - villains) / (heroes + villains + 1)
        _dbg(f"detect_devil_angel_shift | shift={shift}")
        return {
            'devil_angel_shift': shift,
            'num_heroes':        self.narrative_roles.get('hero_count', 0),
            'num_villains':      self.narrative_roles.get('villain_count', 0),
        }

    # ==================== Stakeholder's claim ====================

    def detect_interactivity(self, batch_delay: float = 0.02,
                              max_retries: int = 3) -> Dict:
        empty = {
            'interactive': 0, 'interaction_cues': [], 'interactive_ratio': 0.0,
            'num_interactive_sentences': 0, 'total_sentences': 0,
            'total_citations': 0, 'sentence_breakdown': [],
        }
        if not self.text:
            return empty

        sentences = re.split(r'(?<=[.!?])\s+', self.text)
        sentences = [s.strip() for s in sentences if s.strip()]
        _dbg(f"detect_interactivity | {len(sentences)} sentences to process")
        if not sentences:
            return empty

        results = self._run_async(
            self._interactivity_async(sentences, batch_delay, max_retries)
        )

        interactive_sentences, all_cues = [], []
        for result, sent in results:
            if result and result.get('is_interactive', False):
                citations = result.get('citation_phrases', [])
                interactive_sentences.append({
                    "sentence": sent,
                    "citation_phrases": citations,
                    "num_citations": len(citations),
                })
                all_cues.extend(citations)

        n = len(interactive_sentences)
        total = len(sentences)
        _dbg(f"detect_interactivity | {n}/{total} interactive")
        return {
            'interactive':               1 if n > 0 else 0,
            'interaction_cues':          all_cues,
            'interactive_ratio':         round(n / total, 3) if total else 0.0,
            'num_interactive_sentences': n,
            'total_sentences':           total,
            'total_citations':           len(all_cues),
            'sentence_breakdown':        interactive_sentences,
        }

    async def _interactivity_async(self, sentences: List[str],
                                    batch_delay: float, max_retries: int):
        async with AsyncOpenAI() as client:
            semaphore = asyncio.Semaphore(self.max_concurrent)

            async def process_one(sent):
                async with semaphore:
                    cache_key = f"interactivity_sent:{sent[:200]}"
                    if cache_key in self.llm_cache:
                        _dbg(f"detect_interactivity | cache hit: {sent[:40]!r}")
                        return self.llm_cache[cache_key], sent

                    system = "You identify citations in sentences. Respond in JSON format."
                    user = f"""
Does this sentence REFERENCE or QUOTE other people's viewpoints?

INTERACTIVE: attributions ("X says"), citations ("According to..."), references ("The CEO stated...")
NON-INTERACTIVE: plain facts with no source attribution.

Sentence: "{sent}"

Return JSON:
{{
  "is_interactive": true/false,
  "citation_phrases": ["all citation phrases found, empty if none"]
}}
"""
                    for attempt in range(max_retries):
                        result = await self._llm_call_async(
                            client, "detect_interactivity", system, user, attempt
                        )
                        if result is not None:
                            self.llm_cache[cache_key] = result
                            await asyncio.sleep(batch_delay)
                            return result, sent
                        await asyncio.sleep(attempt + 1)

                    _dbg(f"detect_interactivity | all retries exhausted: {sent[:40]!r}")
                    return {"is_interactive": False, "citation_phrases": []}, sent

            return await asyncio.gather(*[process_one(s) for s in sentences])

    # ==================== Text Polarization ====================

    async def _detect_text_polarization_async(self, batch_delay: float,
                                               max_retries: int) -> Dict:
        empty = {'stakeholders': [], 'sentence_polarization': [],
                 'polarization_summary': {}, 'polarization_bias': 0.0}
        if not self.text:
            return empty

        key = f"polarization:{self.text[:200]}"
        if key in self.llm_cache:
            _dbg("detect_text_polarization | cache hit")
            return self.llm_cache[key]

        _dbg(f"detect_text_polarization | text length={len(self.text)}")

        system = ("You are a stance analysis expert specializing in identifying "
                  "stakeholder positions and bias in text.")
        user = f"""
Analyze stakeholder-based polarization in the text.

STEP 1: IDENTIFY STAKEHOLDERS (political parties, orgs, demographic groups,
        industries, ideological groups)
STEP 2: For each sentence, determine stance toward EACH stakeholder:
- "pro": supports/favors/praises this stakeholder
- "con": opposes/criticizes/harms this stakeholder
- "neutral": mentions without clear stance
- "n/a": sentence doesn't relate to this stakeholder

Text:
\"\"\"{self.text}\"\"\"

Output STRICT JSON:
{{
  "stakeholders": ["stakeholder1", "stakeholder2"],
  "sentence_stances": [
    {{
      "sentence": "exact sentence",
      "stances": {{
        "stakeholder1": "pro|con|neutral|n/a",
        "stakeholder2": "pro|con|neutral|n/a"
      }}
    }}
  ]
}}
Identify AT LEAST 2 stakeholders for any contentious topic.
"""
        async with AsyncOpenAI() as client:
            for attempt in range(max_retries):
                data = await self._llm_call_async(
                    client, "detect_text_polarization", system, user, attempt
                )
                if data is not None:
                    stakeholders = data.get("stakeholders", [])
                    sentence_stances = data.get("sentence_stances", [])

                    summary = {s: {'pro': 0, 'con': 0, 'neutral': 0}
                               for s in stakeholders}
                    for entry in sentence_stances:
                        for s, label in entry.get("stances", {}).items():
                            if s in summary and label in summary[s]:
                                summary[s][label] += 1

                    biases = []
                    for counts in summary.values():
                        pro, con = counts['pro'], counts['con']
                        if pro + con > 0:
                            biases.append(abs(pro - con) / (pro + con))

                    out = {
                        'stakeholders':          stakeholders,
                        'sentence_polarization': sentence_stances,
                        'polarization_summary':  summary,
                        'polarization_bias':     sum(biases) / len(biases) if biases else 0.0,
                    }
                    self.llm_cache[key] = out
                    await asyncio.sleep(batch_delay)
                    _dbg(f"detect_text_polarization | stakeholders={stakeholders} "
                         f"bias={out['polarization_bias']:.3f}")
                    return out
                await asyncio.sleep(attempt + 1)

        _dbg("detect_text_polarization | all retries failed")
        return empty

    def detect_text_polarization(self, batch_delay: float = 0.02,
                                  max_retries: int = 3) -> Dict:
        return self._run_async(
            self._detect_text_polarization_async(batch_delay, max_retries)
        )

    # ==================== Get All Metrics ====================

    def get_all_metrics(self, batch_delay: float = 0.02) -> Dict:
        print("\n" + "="*60)
        print("Starting analysis with async optimization...")
        print("="*60)

        start_time = time.time()

        results = {
            'balanced_pro_con':  self.balanced_pro_con(batch_delay),
            'narrative_roles':   self.detect_narrative_roles(batch_delay),
            'harm_index':        self.detect_harm_index(batch_delay),
            'devil_angel_shift': self.detect_devil_angel_shift(),
            'interactivity':     self.detect_interactivity(batch_delay),
            'text_polarization': self.detect_text_polarization(batch_delay),
        }

        elapsed = time.time() - start_time
        self._save_cache()

        print("="*60)
        print(f"✓ Analysis complete in {elapsed:.2f} seconds")
        print(f"✓ Cache now contains {len(self.llm_cache)} entries")
        print("="*60 + "\n")

        return results

    # ==================== Utility Methods ====================

    def clear_cache(self):
        self.llm_cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        _dbg("Cache cleared.")

    def get_cache_stats(self) -> Dict:
        return {
            'total_entries': len(self.llm_cache),
            'cache_file':    str(self.cache_file),
            'cache_exists':  self.cache_file.exists(),
            'cache_size_kb': (self.cache_file.stat().st_size / 1024
                              if self.cache_file.exists() else 0),
        }
