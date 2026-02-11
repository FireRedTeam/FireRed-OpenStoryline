Now you need to extract/select synthesis parameters for the TTS provider **"{{provider_name}}"** based on the user request.

【User Request】
{{user_request}}

【Available Parameter Definitions (only these fields are allowed)】
{{schema_text}}

## Output Requirements

1. Output **JSON object (dict) only** — no markdown, no extra text.
2. You may output **only** the fields defined in the available parameter definitions; do not invent fields.
3. Values must match the specified `type`:

   * `"int"` / `"float"`: output a numeric value
   * `"str"`: output a string
   * `"bool"`: output `true` / `false`
4. `enum` : you must choose **one** value from the list that best matches the user request.
5. `range`:  output a number **within the range** (you may keep 1 decimal place).
6. Fields not mentioned by the user may be omitted; but if the user explicitly asks for something (e.g., gender/voice, speaking rate, volume), **try to output the corresponding fields**.
