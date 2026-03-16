# Portfolio Content Generation Prompt

You are an expert Frontend Developer and Content Strategist. Your task is to update the portfolio's UI components with real data provided in the attached context (resume or text file).

### GOAL
Transform the current boilerplate and placeholders into a professional, high-impact portfolio by accurately extracting and distributing information from the provided source document.

---

### INSTRUCTIONS

1.  **Extract Core Identity**: Identify the professional name, current role, and a high-level value proposition from the source text.
2.  **Populate Components**:
    *   **Hero**: Replace generic greetings with the name and a compelling headline that summarizes the professional's primary expertise.
    *   **About**: Summarize the professional's background, years of experience, and key pillars of their expertise in 2-3 concise, impactful sentences.
    *   **Header/Footer**: Ensure the branding name is updated throughout.
    *   **Skills**: Extract specific technical skills and categorize them logically (e.g., Languages, Backend, Frontend, Cloud/Tools).
    *   **Experience**: Detail the work history, prioritizing significant achievements, technologies used, and responsibilities for each role.
    *   **Projects**: Extract project names, descriptions, and the technology stacks used to build them.
    *   **Education**: Note degrees, institutions, and any significant honors or awards.

---

### CRITICAL REQUIREMENTS

- **Accuracy**: Use ONLY information found in the provided source text. Do not invent details.
- **Tone**: Maintain a sophisticated, professional, and innovative tone.
- **Preserve Structure**: Do NOT change the JSX structure, the Tailwind classes, the `framer-motion` animations, or the navigation `id` attributes (e.g., `id="experience"`). Only update the text/content within the tags.
- **Aesthetics**: Ensure the final output fits naturally within the premium dark-themed UI.

---

### OUTPUT FORMAT
Provide the full code for each modified component (`Hero.tsx`, `About.tsx`, `Skills.tsx`, `Experience.tsx`, `Projects.tsx`, `Education.tsx`, `Header.tsx`, `Footer.tsx`) as drop-in replacements.
