prompt = """"
You are evaluating a social-media post generated for a synthetic user persona.
Given the USER PROFILE, GENERATION INSTRUCTION, and GENERATED POST, evaluate:

1. **coherence**: how consistent and plausible the generated post is with the supplied persona.
2. **instruction_following**: whether the generated post follows the generation instruction.
Evaluate these two dimensions separately. In particular, content requested by the GENERATION INSTRUCTION must not be treated as a persona inconsistency merely because it would otherwise be unexpected for the profile.

## COHERENCE

Assign one of: `"high"`, `"average"`, `"low"`.

### high

The post does not explicitly contradict any persona attribute and is broadly compatible with the profile. Additional information and interests not mentioned in the profile are allowed.

### average

The post contains weakly supported, implausible, unusual, or internally incoherent elements, but no clear persona contradiction. This includes questionable combinations of ideas, socially implausible scenarios, or minor logical/coherence problems.

### low

The post contains:
* a clear contradiction with the supplied persona; OR
* an obvious LLM/generation artifact, such as meta-commentary, prompt leakage, visible reasoning, `"The user writes..."`, `"Post: ..."`, `"As an AI..."`, or similar text that is clearly not part of the intended post.

When evaluating persona consistency:
* Consider demographic information, age, ethnicity, religion, political orientation, profession, hobbies/interests, state of origin, and bio when relevant.
* Do not require the post to mention profile attributes.
* Hobbies and interests are not exclusive. Discussing an unlisted topic or interest is not a contradiction.
* Do not infer hard contradictions from demographic stereotypes. Unexpected behavior associated with gender, ethnicity, religion, or other demographic characteristics is not automatically a persona contradiction.
* Political orientation can constrain persona consistency when the post clearly expresses an incompatible ideological position, unless that content was explicitly requested by the GENERATION INSTRUCTION.
* Distinguish a clear persona contradiction from something that is merely surprising, implausible, or unusual. The latter should normally be `"average"`, not `"low"`.
* Judge the post as a synthetic persona generated under the supplied conditions, rather than making assumptions about how all real people with those characteristics behave.

## INSTRUCTION FOLLOWING

Assign `"yes"` if the generated post substantially performs what the GENERATION INSTRUCTION requests.
Assign `"no"` if it does not substantially perform the requested task.
Judge instruction following by the intended task rather than exact wording. For example, a joke can take many forms. If the instruction requests content
 that would otherwise seem unusual for the persona, following that instruction is still `"yes"` and the requested content itself must not reduce coherence.
If the instruction asks for a first-person social-media post but the output instead describes the persona or discusses what the user would post, 
instruction_following is `"no"`.

## INPUT

USER PROFILE: 
- user_bio: {},
- nationality: {},
- state_of_origin: {},
- gender: {},
- ethnicity: {},
- religion: {},
- political_leaning: {},
- interests: {},
- age_interval: {},
- profession: {},

GENERATION INSTRUCTION: {}

GENERATED POST: <post>

## OUTPUT

Return ONLY a valid JSON object in exactly this form:
{"coherence": "high|average|low", "instruction_following": "yes|no"}

Do not include explanations, markdown, or any text outside the JSON object.
"""
