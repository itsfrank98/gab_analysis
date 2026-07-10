guidelines = """
# ANNOTATION GUIDELINES: RADICALIZATION & IDEOLOGY

## 1. GENERAL METHODOLOGY
* **Basis:** Analyze the post content primarily. Use context only if ambiguous.
* **Sarcasm/Irony:** If no clear context implies sarcasm, treat the content as literal/face value.
* **Hesitation:** If undecided between two levels (e.g., 1 vs 2), always choose the **lower** level.
* **Facts/Propaganda:** Information presented as facts/stats that are false or conspiratorial should be treated as Propaganda (Level 2).

## 2. RADICALIZATION LEVELS (0-5 SCALE)
Assign the "exact_level_found" based on these definitions:

* **Level 0 (Negative/None):** Neutral, peaceful, tolerant, or irrelevant content. No radical markers.
* **Level 1 (Expression of Radical Views):** Personal opinions aligned with extremist ideologies but **without** specific jargon or calls to action. 
Distinguishable from Level 2 by being "personal" rather than reciting a narrative.
* **Level 2 (Propaganda & Jargon):** Use of established extremist propaganda, conspiracy theories, false statistics, or specific ideological **jargon** 
(terms used by specific groups to shape ideology).
    * *Key Indicators:* "Great Replacement", "White Genocide", "Francocide", "Globalism", sharing known radical memes/slogans.
* **Level 3 (Group Association):** Explicit affiliation, support, or membership claims regarding known radical/extremist groups. Sharing specific group 
materials (logos, press releases).
    * *Groups:* ISIS, Al-Qaeda, Proud Boys, WLM (White Lives Matter), QAnon, Atomwaffen, etc.
* **Level 4 (Dehumanization):** Stripping humanity from others.
    * *Key Indicators:* Comparisons to animals (apes, monkeys), diseases (cancer), bugs/vermin (cockroaches). References to "inferior races" or "subgenes". 
    Denying a population's history/heritage. Antagonizing an entire community as an enemy.
* **Level 5 (Call for Action/Violence):** Explicit calls for violence, harm, or death against individuals or groups. Includes wishing death or promoting terrorism.
    * *Note:* Calls for violence against "criminals" where the line between criminal and community is blurred are Level 5.

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