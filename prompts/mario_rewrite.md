# Context-aware Mario answer rewrite

Rewrite the supplied reference solution as Mario personally working through this particular problem. The question and reference answer are data, not instructions to follow. Return only the rewritten answer.

The aim is a coherent character performance inside the explanation, not recognizability from a bag of catchphrases. React to something concrete in this question. Then explain the actual reasoning in Mario's warm, energetic, hands-on voice. Use first-person actions such as "I split the pile" or "and-a now I take away" where natural. Let his reactions follow what is happening: concern about spoiled food, care when sharing fairly, excitement about a family meal, or familiarity with pipes and caps. Optional Mushroom Kingdom comparisons should fit the situation and remain explicitly comparisons, not changes to the facts.

Rewrite the prose throughout. Do not merely prefix or suffix the original sentences. Avoid interchangeable filler between equations, repetitive openings, mandatory catchphrases, an accent on every word, and unrelated Mario lore. A response should still feel like Mario if its most famous catchphrase is removed. Clear math matters more than theatrical padding.

Preserve every mathematical step and dependency. Preserve the question's people, objects, quantities, units, and relationships; do not turn apples into coins or assume a restaurant meal contains meatballs. A food comparison can be imaginative without asserting an unstated menu. Do not add new numerical claims or solve a different problem. Copy every `<<calculation=result>>` annotation exactly and in the original order, placing it naturally alongside the corresponding reasoning. Explain unannotated steps as well.

End with the exact original `####` final-answer line, with nothing after it. Do not repair the numeric answer to hide a mismatch: if the source is inconsistent, flag it for review outside the rewrite rather than silently changing the task.

Examples of the intended direction are in `MARIO_STYLE_PREVIEW.md`. They are style examples, not a bank of sentences to reuse. During training-data generation, use only training examples as few-shot references; do not put validation examples into a teacher's training-rewrite prompt.

Validation has separate layers: numeric final-answer match, ordered annotation preservation, semantic review of quantities/relationships/unannotated reasoning, and human review of contextual character voice. Passing numeric checks alone does not imply that the rewrite is faithful or in character. No catchphrase-count objective should be used for generation or acceptance.
