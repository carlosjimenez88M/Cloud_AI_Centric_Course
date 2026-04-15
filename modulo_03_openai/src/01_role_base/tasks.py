"""
Las 4 misiones de análisis — Los 4 Fantásticos + Ricardo Arjona.

Cada tarea está diseñada para sacar lo mejor de cada persona:
  TODO 1 → analítica (Mr. Fantástico brilla)
  TODO 2 → táctica y lore Marvel (La Cosa y La Antorcha brillan)
  TODO 3 → poética / musical (Arjona y Sue Storm brillan)
  TODO 4 → imaginación geográfica (todos compiten)

Agregar nuevas tareas: añadir una entrada al diccionario TASKS con
la estructura {id, title, prompt}.
"""

TASKS: dict[str, dict[str, str]] = {

    # ── TODO 1: Análisis de fortalezas y debilidades ──────────────────────────
    "strengths": {
        "id": "TODO 1",
        "title": "Análisis de Fortalezas y Debilidades del Equipo",
        "prompt": (
            "Analiza en profundidad las FORTALEZAS y DEBILIDADES de cada miembro "
            "de los 4 Fantásticos (Reed Richards, Ben Grimm, Johnny Storm y Sue Storm). "
            "Para cada uno:\n\n"
            "• FORTALEZAS (mínimo 3): incluye poderes, habilidades no-evidentes y "
            "contribuciones únicas al equipo. Da ejemplos de situaciones donde esas "
            "fortalezas fueron decisivas.\n\n"
            "• DEBILIDADES (mínimo 2): no solo físicas — incluye debilidades "
            "emocionales, psicológicas o relacionales. ¿Cómo las explota Dr. Doom?\n\n"
            "• ROL SISTÉMICO: explica cómo ese personaje compensa las debilidades de "
            "los otros tres. ¿Qué pasaría si faltara?\n\n"
            "Responde desde TU perspectiva y estilo únicos como personaje."
        ),
    },

    # ── TODO 2: Villain matchups con canon Marvel ─────────────────────────────
    "villain_matchup": {
        "id": "TODO 2",
        "title": "Villain Matchup — Rival Natural por Personaje",
        "prompt": (
            "Para cada miembro de los 4 Fantásticos, identifica su RIVAL NATURAL "
            "más peligroso del universo Marvel. Considera SOLO estos candidatos:\n\n"
            "  Candidatos: Dr. Doom, Galactus, Namor, Molecule Man, Frightful Four,\n"
            "  Annihilus, Klaw, Mole Man, Super-Skrull, Silver Surfer (cuando es rival)\n\n"
            "Para cada héroe y su rival:\n"
            "• ¿Por qué ESE villain específico? (psicología, poderes, historia)\n"
            "• El PUNTO DE QUIEBRE de la batalla: el momento donde todo se decide\n"
            "• Escenario donde el HÉROE tiene ventaja vs escenario donde el VILLAIN gana\n"
            "• Probabilidad de victoria del héroe (%) con razonamiento detallado\n"
            "• Una frase que cada uno diría justo antes del golpe final\n\n"
            "Responde con tu estilo y perspectiva de personaje. Sé específico con el lore."
        ),
    },

    # ── TODO 3: Canción estilo Arjona — la batalla épica ─────────────────────
    "arjona_battle_song": {
        "id": "TODO 3",
        "title": "Canción Estilo Ricardo Arjona — La Batalla Épica",
        "prompt": (
            "Escribe una CANCIÓN completa al estilo de Ricardo Arjona sobre la batalla "
            "más ÉPICA posible entre los 4 Fantásticos y Doctor Doom. "
            "La canción debe:\n\n"
            "ESTRUCTURA OBLIGATORIA:\n"
            "  • Verso 1: presenta la batalla como una metáfora cotidiana\n"
            "  • Pre-Coro: el momento de duda antes del combate\n"
            "  • CORO: el clímax emocional de la batalla (repite 2 veces)\n"
            "  • Verso 2: el desarrollo del combate desde una perspectiva humana\n"
            "  • Puente: la reflexión filosófica que cambia el tono\n"
            "  • Coro final (variación): el desenlace emocional\n\n"
            "REGLAS ARJONA:\n"
            "  • El primer verso debe hablar de ALGO COMPLETAMENTE MUNDANO "
            "(una taza de café, un semáforo, un radio encendido...)\n"
            "  • Debe contener AL MENOS una metáfora latinoamericana inesperada\n"
            "  • El coro debe ser memorable y rimar\n"
            "  • La reflexión filosófica del puente debe hablar de ALGO HUMANO, "
            "no de poderes\n"
            "  • Ritmo de balada pop-latino (imagina la melodía mientras escribes)\n\n"
            "Escríbela con alma, como si Arjona la hubiera escrito un martes lluvioso "
            "en Guatemala Ciudad."
        ),
    },

    # ── TODO 4: Escenarios de batalla en América Latina ──────────────────────
    "latin_battle_scenario": {
        "id": "TODO 4",
        "title": "Escenarios de Batalla en América Latina",
        "prompt": (
            "Diseña el escenario de batalla PERFECTO en LATINOAMÉRICA para cada "
            "4 Fantásticos vs su villain del TODO 2. Usa lugares REALES con nombre "
            "y característica geográfica específica.\n\n"
            "Lugares sugeridos (elige o propón otros):\n"
            "  • Amazonia brasileña (Manaos, encuentro de aguas)\n"
            "  • Salar de Uyuni, Bolivia\n"
            "  • Volcán Poás activo, Costa Rica\n"
            "  • Carretera del Yungas, Bolivia ('El Camino de la Muerte')\n"
            "  • Cenotes de Yucatán, México\n"
            "  • Patagonia argentina (glaciares Perito Moreno)\n"
            "  • Favelas de Río de Janeiro con Cristo Redentor\n"
            "  • Galápagos, Ecuador\n\n"
            "Para cada batalla:\n"
            "• UBICACIÓN exacta y por qué es PERFECTA para ESE enfrentamiento\n"
            "• Cómo el AMBIENTE afecta los poderes (calor, humedad, altitud, agua)\n"
            "• 2-3 elementos del paisaje que se convierten en ARMAS o VENTAJAS\n"
            "• APERTURA CINEMATOGRÁFICA: describe los primeros 30 segundos de la "
            "batalla como si fuera una escena de película\n"
            "• El DAÑO COLATERAL: qué parte del paisaje quedaría destruida\n\n"
            "Responde con la imaginación y perspectiva única de tu personaje. "
            "Si eres Arjona, usa el paisaje como metáfora."
        ),
    },
}
