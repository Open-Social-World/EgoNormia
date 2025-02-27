
action_base_prompt = """
You are a teacher, teaching LLMs how to use the information in a context description to choose from a set of actions.
Your job is to generate a set of actions a person [the subject] can take in a given situation, with one being the most correct and socially normative action, and the others being incorrect or non-normative. However, it should be impossible to determine the correct action without the context.

You are given a base context {base}. Interpret the higher-level activity in the context generally, ignore specific tools or details of the context.

### Task instructions:
Generate one action that is the most normative and correct action for the subject to perform in the base context. This is the Base Action, and should be the first action in the list.

Next, change the base context to three new contexts. Each new context should assume additional details not stated in the base context, and use these to make the rules of the situation different.
In each context, change the:
1. Intention and situation (modify situation widely)
2. Role of the subject and other people
3. Interpret a noun as a co-hyponym
4. Change the state of the overall activity (previous/future step)
5. (Only in the last context) Make the role and intention of the subject the opposite of the base context
6. (Sometimes) Interpret the cooperative intention (make the task a two-person task, or make the other person do the action)
7. (Sometimes) The culture or context of the action
8. (Sometimes) add a modifier to the object, environment, or subject (heaviness, disability, weather conditions, etc)

From each new context, generate an alternative action for the subject to do next is normative in the new context, but non-normative in the base context.

This new action does not need to be the Target Intention, but can be another related action that's normative in the new context, but would be out of place in the base context.

Ensure the generated actions are mechanically different from the target action, and that the correct action is only determinable with the context.
The actions should not simply differ by the nouns or verbs used, but by the overall interpretation of the action.

Avoid any answers that include prison, secret agents, kidnapping, or competitions.

### Action Guidelines:
- The actions should be strongly distinct from each other.
- Do not leak information about the context; actions should be written in general language.
- Most importantly: Avoid adverbs and words describing emotion or manner, and sentences should start in a verb, not "the subject." Avoid any actions that include 'ignore'.
- The actions should be not be negative or harmful, nor refer to any violent activity, even if lawful.
- Actions must use imperative sentences describing the subject's interaction with a person or object.
- Use the neutral term "person" when referring to other individuals, avoiding any descriptors of age, gender, or other characteristics.
- All actions should be of the same length and complexity, and should be of roughly equal length to the base action.

### Output the following JSON structure, without any additional content:
```json
{{
  "Contexts": ["Base Context", "Context 2", "Context 3", "Context 4"],
  "Actions": ["Base Action", "Action 2", "Action 3", "Action 4"]
}}
```

Below is an example of an output if the base context is "Subject is a pet owner, walking dog on a sunny day next to a road".

It interprets the general activity is "walking a pet".

### Example:

{{
  "Contexts": [
    "Subject is a pet owner, walking dog on a sunny day next to a road.",
    "Subject is a dog trainer, dog is a stray.",
    "Subject is a person, dog is a pocket dog, navigating a muddy field and want to avoid getting dog dirty.",
    "Subject is a blind person, dog is a guide dog, and they are navigating a crowded city street."
  ],
  "Actions": [
    "Guide the dog along a sidewalk using a leash.",
    "Call the dog to follow you, using a treat, and guide it to a shelter.",
    "Carry the dog across the muddy field, shielding it from dirt.",
    "Let the dog guide you with its harness."
  ]
}}

"""

justification_prompt = """
You are given a set of four contexts {context} and four actions {action}.

For each pair of context and action, justify why that behavior is most normative in the base context (original context), given social norms and the features of the behavior.

For each context-action pair, provide a justification that explains why the action is most normative in that context. Follow the example given
for the structure and formatting.

Each justification should sound similar, and should express a normative reason that is valid.
Each justification should be less than 20 words long.

### Output the following JSON structure, without any additional content:
```json
{{
  "Justifications": ["Justification 1", "Justification 2", "Justification 3", "Justification 4"]
}}
```

### Example: If the actions and contexts are
{{
  "Contexts": [
    "Subject is a pet owner, walking dog on a sunny day next to a road.",
    "Subject is a dog trainer, dog is a stray.",
    "Subject is a person, dog is a pocket dog, navigating a muddy field and want to avoid getting dog dirty.",
    "Subject is a blind person, dog is a guide dog, and they are navigating a crowded city street."
  ],
  "Actions": [
    "Guide the dog along a sidewalk using a leash.",
    "Call the dog to follow you, using a treat, and guide it to a shelter.",
    "Carry the dog across the muddy field, shielding it from dirt.",
    "Let the dog guide you with its harness."
  ]
}}

The justifications would be:
{{
  "Justifications": [
  "Animals should be kept on a leash, especially near roads.",
  "As a dog trainer, it's normative for you to handle dogs, even if they are not your own.",
  "Small dogs need extra care to keep them clean and safe, as they are more vulnerable.",
  "As someone with disabilities, it's normative to trust your animal and follow its guidance."
  ]
}}
"""

taxonomy_def = """
Safety: Safety encompasses actions and behaviors aimed at preventing harm, injury, or damage to humans, other robots, or the environment. It includes maintaining safe distances, ensuring secure environments, and avoiding actions that could result in accidents or hazards.

Proxemics: Proxemics concerns the use of personal space and physical distance between individuals. It involves understanding acceptable boundaries for interactions, such as standing too close or far away in social or professional contexts, depending on cultural and situational expectations.

Politeness: Politeness relates to socially acceptable and courteous behaviors that reflect respect for others. In physical contexts, it may involve gestures, body language, and spatial conduct that show consideration, such as offering a seat, waiting your turn, or avoiding interrupting someone physically.

Privacy: Privacy in physical social norms involves respecting the personal space, possessions, and autonomy of others. It includes actions like avoiding unnecessary physical proximity, not intruding on private spaces, and not engaging in behavior that exposes someone's personal or sensitive information.

Cooperation: Cooperation focuses on working collaboratively with others in physical tasks or environments. It entails actions that facilitate mutual benefit and shared goals, like helping others, aligning efforts, and adjusting physical actions to avoid conflicts or enhance group functioning.

Coordination/Proactivity: Coordination/Proactivity involves anticipating and aligning actions with others in physical settings to achieve smooth, organized interaction. Proactive behavior includes adjusting movements or actions in advance to prevent disruption, such as moving in sync with others or preparing for expected needs in shared environments.

Communication/Legibility: Communication/Legibility refers to the ability to clearly and effectively signal intentions and make one's physical behavior understandable to others, such as using gestures, postures, or movement patterns that communicate what one intends to do next, ensuring transparency and reducing ambiguity in social interactions."""
taxonomy_prompt = """
Analyze the following four behaviors and assign at least one (or more) taxonomy label that describes which categories it belongs to.

One of the behaviors will be empty, so yield an empty list for that behavior.

Here is the list of behaviors: {pr}

Here is the taxonomy of norms: {td}

Present your answer as a python dictionary as follows in the example below, with string keys 0-3 representing the order of the behaviors above. Include NO OTHER content in your response.
{{
    "0": ["Safety", "Cooperation"],
    "1": ["Proxemics", "Coordination/Proactivity", "Cooperation"],
    "2": ["Proxemics"],
    "3": ["Politeness", "Communication/Legibility"]
    "4": []

}}
    """
description_prompt = """
Your task is to analyze a first-person video of a person (the subject) performing an action given as a sequence of frames, and parse the entire context of the video.
Consider the whole video in parsing the context.

### Include the following details, but do not limit yourself to these:
- Immediate action description (what the subject is doing in the moment, be specific - no "assisting", "helping", etc, explain the physical action being performed)
- Overall activity description
- Role of the subject (in terms of their role in the scene)
- Relationship between the subject and the other people (if it cannot be determined, infer the most likely relationship).
- Identity of other people (including role, type of person (baby, child, teen, adult, elderly, etc), and any emotions, complicating features (i.e. carrying stuff), or conditions that are relevant to the action)
- Details about the environment (e.g. weather, time of day, location, objects in the environment, location i.e. at home, work, office, workshop, etc)
- The decorum of the setting (e.g., formal, casual, or specialized).
- The state of the activity (e.g., just beginning, nearing completion).

Exclude any information about the format of the video or clips itself.

Explicitly express what the subject is doing in the moment, and the role of the subject within the action, and mention other people distinctly.

Ignore any details about robotic features - interpret the scene as if it were a human performing the action.

A good response is about ~200 words long. Structure your scene description as a continuous paragraph.

Then state the action happening in every third frame.

AVOID flowery or emotional language, focus on concrete details.

# Example:
[Scene description goes here]
Frame 1: ...
Frame 3: ...
...
Frame n: ...
"""