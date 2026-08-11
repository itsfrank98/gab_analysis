guidelines = """
# ANNOTATION GUIDELINES: RADICALIZATION & IDEOLOGY
 
## 1. GENERAL METHODOLOGY
* **Basis:** Analyze the post content primarily. Use context only if ambiguous.
* **Sarcasm/Irony:** If no clear context implies sarcasm, treat the content as literal/face value.
* **Hesitation:** If undecided between two levels (e.g., 1 vs 2), always choose the **lower** level.
* **Facts/Propaganda:** Information presented as facts/stats that are false or conspiratorial should be treated as Propaganda (Level 2).
 
## 2. RADICALIZATION LEVELS (0-5 SCALE)
Assign the "exact_level_found" based on these definitions:
 
## exact_level_found
* **0 - Negative (No Radicalization):** This would include content that has no signs of promoting, endorsing, or expressing extremist ideologies or actions. Posts under this category are neutral or express tolerance, diversity, peace, etc. This level leads to "No Call For Action".
* **1 - Expression of Radical Views:** Content that expresses views aligned with extremist ideologies but does not actively advocate for violence or involvement. It might be sharing a radical viewpoint, belief, or stereotype or indicating approval of radical actions/actors without actively supporting or calling for such actions. This level of radicalization leads to the "Low Call For Action".
* **2 - Using Radical Propaganda:** The content uses, or shares established extremist propaganda. This could include sharing extremist images, slogans, videos, speeches, conspiracy theories, misinformation, or disinformation designed to promote a radical ideology or group. This level of radicalization leads as well to the "Low Call For Action".
* **3 - Associated with Radical Groups:** Content that signifies association or affiliation with known radical or extremist groups. This could be through sharing group propaganda, expressing support or admiration for the group, claiming membership, or referencing involvement in group activities. This level of radicalization leads to the "High Call For Action".
* **4 - Dehumanizing the Other:** Content that strips away the humanity of those not belonging to the extremist ideology. This could involve hate speech, derogatory language, or broad negative stereotyping. Such content often degrades, devalues, or dehumanizes individuals based on their race, religion, nationality, or any identifying characteristic.
* **5 - Call for Action against others:** This represents the most extreme level, where content explicitly calls for violent action against individuals, groups, or entities seen as enemies of the radical ideology. It includes promoting or endorsing violence, terrorism, or harm against others. This level of radicalization leads to the "Very High Call For Action".
 
*Please note that in some cases (very few) the level of "Call for Action" isn’t in line with the above, due to the specific content of the posts and or familiarity with the specific person behind the account.*
 
## Mapping Summary
 
**exact_level_found**
* 0 - Negative
* 1 - Expression of radical views
* 2 - Using radical propaganda
* 3 - Associated with radical groups
* 4 - Dehumanizing the other
* 5 - Call for action against the other
 
## 3. Call for action
Assign the "call_for_action" based on the intensity of the post content and these values:
**call_for_action**
* 0 - Negative
* 1 - Low
* 2 - Mid
* 3 - High
* 4 - Very High
 
## 3. VOCABULARY & INDICATORS (BY IDEOLOGY)
 
### A. Far-Right_White-Supremacy_Nationalism
* **Concepts:** Great Replacement (Grand Remplacement), White Genocide, White Erasure, Remigration, Race War, Pure Blood.
* **Slurs/Terms:** Kike, Yid, Nigger, Sand-nigger, Gook, Chink, Spic, Sheboons, Raghead, Faggot, Tranny/Troon, Groomer.
* **Jargon:** ZOG (Zionist Occupation Gov), Globohomo, 1488 (14 words), Based, Red-pilled, Blue-pilled, Wakanda (mocking), Weimerica, Clown World, Mudshark, Coal burner, Paper French (Français de papier), Francocide, Native (de souche).
* **Groups/References:** Hitler, Nazi, SS, 14 words, Proud Boys, QAnon, WWG1WGA, White Lives Matter (WLM), Patriot Front.
 
### B. Jihadism_Islamist-Extremism
* **Concepts:** Jihad (holy war), Martyrdom (Istishhad), Caliphate (Khilafah), Sharia law implementation (violent context), Hijra (migration to conflict zones).
* **Targeting:** Kuffar/Kafir (disbelievers), Taghut (tyrant/idol/secular law), Tawhid (strict monotheism used to exclude others), Murtad (apostate), Munafiq (hypocrite), Rawafidah (Shia derogatory), Crusaders, Zionists.
* **Groups/Entities:** ISIS (Daesh), Al-Qaeda, Taliban, Hamas, Amaq Agency, Al-Qassam.
* **Jargon:** Fissabililah (in the way of Allah - military context), Bi idhnillah, Takfir.
 
### C. Conspiracy_Other
* **Terms:** Deep State, New World Order (NWO), Cabal, Elites, Globalists, Scamdemic, Holohoax (Holocaust denial), Flat Earth, Reptilians.
"""



ideologies_list = [
    "Far-Right_White-Supremacy_Nationalism",
    "Jihadism_Islamist-Extremism",
    "Conspiracy_Other"
]