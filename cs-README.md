# Fei - Pokročilý asistent pro kód 🐉
## Překald: Claude 3.7 sonnet

![Licence](https://img.shields.io/badge/license-MIT-blue.svg)
![Verze](https://img.shields.io/badge/version-0.1.0-green.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Stav](https://img.shields.io/badge/status-early%20development-orange.svg)

> Fei (pojmenovaný po čínském létajícím drakovi přizpůsobivosti) je výkonný asistent pro kód, který kombinuje možnosti umělé inteligence s pokročilými nástroji pro manipulaci s kódem a distribuovaným systémem paměti.

<div align="center">
  <img src="https://raw.githubusercontent.com/david-strejc/fei/refs/heads/main/tmp-logo.jpg" alt="Fei Logo" width="200"/>
</div>

## 📑 Obsah

- [Přehled](#-přehled)
- [Vize projektu](#-vize-projektu)
- [Funkce](#-funkce)
- [Instalace](#-instalace)
- [Použití](#-použití)
- [FEI Network](#-fei-network)
- [Architektura](#-architektura)
- [Dokumentace](#-dokumentace)
- [Známé problémy a chyby](#-známé-problémy-a-chyby)
- [Plán projektu](#-plán-projektu)
- [Přispívání](#-přispívání)
- [Licence](#-licence)

## 🔍 Přehled

Fei je pokročilý asistent pro kód poháněný umělou inteligencí, vytvořený pro zlepšení pracovních postupů při vývoji softwaru. Integruje se s více poskytovateli LLM, nabízí výkonné nástroje pro práci s kódem a obsahuje distribuovaný systém paměti pro trvalé znalosti napříč sezeními.

Využitím schopností velkých jazykových modelů jako Claude a GPT poskytuje Fei inteligentní asistenci pro programovací úkoly, vyhledávání kódu, refaktoring a dokumentaci.

## 🌈 Vize projektu

Projekt Fei představuje více než jen asistenta pro kód; je součástí širší vize demokratizovaného AI ekosystému nazvaného FEI Network (Síť létajícího draka přizpůsobivosti).

Síť FEI si klade za cíl být skutečně demokratickým, distribuovaným systémem umělé inteligence, který slouží kolektivnímu dobru prostřednictvím:

1. **Distribuované zpracování**: Využití kolektivní výpočetní síly napříč miliony individuálních uzlů
2. **Federace specializovaných inteligencí**: Vytvoření sítě specializovaných inteligencí, které spolupracují prostřednictvím otevřených protokolů
3. **Úkolově orientovaný příspěvek**: Účastníci přispívají podle svých schopností, přeměňují výpočetní výkon z plýtvavé soutěže na účelnou spolupráci
4. **Globální začlenění**: Aktivní návrh pro účast napříč ekonomickými, geografickými, lingvistickými a kulturními hranicemi
5. **Orientace na veřejné blaho**: Slouží kolektivním zájmům lidstva spíše než úzkým prioritám

Projekt stojí jako alternativa k centralizovaným přístupům k umělé inteligenci, zaměřuje se na lidskou samostatnost, demokratizaci umělé inteligence a spravedlivé rozdělení výpočetního výkonu.

## ✨ Funkce

### Integrace LLM

- **Podpora více poskytovatelů**: Bezproblémová integrace s Anthropic, OpenAI, Groq a dalšími prostřednictvím LiteLLM
- **Výběr modelu**: Snadné přepínání mezi různými modely LLM
- **Správa API klíčů**: Bezpečné zacházení s API klíči se správnou prioritou

### Nástroje pro manipulaci s kódem

- **Inteligentní vyhledávání**:
  - `GlobTool`: Rychlé porovnávání vzorů souborů pomocí glob vzorů
  - `GrepTool`: Vyhledávání obsahu pomocí regulárních výrazů
  - `SmartSearch`: Kontextově citlivé vyhledávání kódu pro definice a použití

- **Úpravy kódu**:
  - `View`: Prohlížení souborů s omezením řádků a offsetem
  - `Edit`: Přesné úpravy kódu se zachováním kontextu
  - `Replace`: Kompletní nahrazení obsahu souboru
  - `RegexEdit`: Úprava souborů pomocí regex vzorů pro hromadné změny

- **Organizace kódu**:
  - `LS`: Výpis adresáře s filtrováním podle vzorů
  - `BatchGlob`: Vyhledávání více vzorů v jediné operaci
  - `FindInFiles`: Vyhledávání vzorů v konkrétních souborech

### Systém správy paměti

- **Memdir**: Paměťová organizace kompatibilní s Maildir
  - Hierarchická paměť se složkami cur/new/tmp
  - Metadata a značky založené na hlavičkách
  - Sledování stavu pomocí příznaků
  - Pokročilý systém filtrování
  - Správa životního cyklu paměti

- **Memorychain**: Distribuovaný paměťový ledger
  - Řetězec odolný proti manipulaci inspirovaný blockchainem
  - Ověřování paměti založené na konsensu
  - Peer-to-peer komunikace uzlů
  - Sdílený mozek napříč více agenty
  - Monitorování zdraví uzlů a sledování úkolů

### Externí služby (MCP)

- **Brave Search**: Integrace webového vyhledávání pro informace v reálném čase
- **Memory Service**: Znalostní graf pro trvalou paměť
- **Fetch Service**: Načítání URL pro přístup k internetu
- **GitHub Service**: Integrace GitHubu pro správu repozitářů
- **Sequential Thinking**: Služba pro vícekrokové uvažování

## 💻 Instalace

```bash
# Klonování repozitáře
git clone https://github.com/david-strejc/fei.git
cd fei

# Instalace z aktuálního adresáře
pip install -e .

# Nebo instalace přímo z GitHubu
pip install git+https://github.com/david-strejc/fei.git
```

### Požadavky

- Python 3.8 nebo vyšší
- Požadované API klíče (alespoň jeden):
  - `ANTHROPIC_API_KEY`: API klíč Anthropic Claude
  - `OPENAI_API_KEY`: API klíč OpenAI
  - `GROQ_API_KEY`: API klíč Groq
  - `BRAVE_API_KEY`: API klíč Brave Search (pro webové vyhledávání)

## 🚀 Použití

### Základní použití

```bash
# Spuštění interaktivního chatu (tradiční CLI)
fei

# Spuštění moderního textového rozhraní založeného na Textual
fei --textual

# Odeslání jedné zprávy a ukončení
fei --message "Najdi všechny Python soubory v aktuálním adresáři"

# Použití konkrétního modelu
fei --model claude-3-7-sonnet-20250219

# Použití konkrétního poskytovatele
fei --provider openai --model gpt-4o

# Povolení protokolování ladění
fei --debug
```

### Python API

```python
from fei.core.assistant import Assistant

# Vytvoření asistenta
assistant = Assistant()

# Jednoduchá interakce
response = assistant.ask("Jaké soubory obsahují funkci 'process_data'?")
print(response)

# Interaktivní sezení
assistant.start_interactive_session()
```

### Proměnné prostředí

Konfigurace Fei prostřednictvím proměnných prostředí:

```bash
# API klíče
export ANTHROPIC_API_KEY=váš_anthropic_api_klíč
export OPENAI_API_KEY=váš_openai_api_klíč
export GROQ_API_KEY=váš_groq_api_klíč
export BRAVE_API_KEY=váš_brave_api_klíč

# Konfigurace
export FEI_LOG_LEVEL=DEBUG
export FEI_LOG_FILE=/cesta/k/souboru.log
```

### Použití paměťového systému

```bash
# Vytvoření struktury paměťového adresáře
python -m memdir_tools

# Vytvoření nové paměti
python -m memdir_tools create --subject "Poznámky ze schůzky" --tags "poznámky,schůzka" --content "Body k diskusi..."

# Seznam pamětí ve složce
python -m memdir_tools list --folder ".Projects/Python"

# Vyhledávání v pamětech
python -m memdir_tools search "python"

# Pokročilé vyhledávání se složitým dotazem
python -m memdir_tools search "tags:python,important date>now-7d Status=active sort:date" --format compact
```

## 🌐 FEI Network

Fei je součástí širší vize sítě FEI - distribuovaného, demokratického systému pro kolektivní inteligenci. Síť funguje jako živá, adaptivní neuronová síť složená z tisíců jednotlivých uzlů, z nichž každý má specializované schopnosti.

### Základní principy sítě

1. **Radikální otevřenost**: Každý s výpočetními zdroji se může účastnit
2. **Emergentní specializace**: Uzly se přirozeně specializují na základě svých schopností
3. **Autonomní organizace**: Síť se sama organizuje prostřednictvím rozhodování založeného na kvóru
4. **Vzájemná hodnota**: Příspěvky jsou spravedlivě odměňovány pomocí FeiCoin
5. **Distribuovaná odolnost**: Bez jediného bodu selhání nebo kontroly

### Specializace uzlů

Síť FEI se skládá ze specializovaných typů uzlů:

- **Matematické uzly**: Řešení komplexních výpočetních problémů a formálního uvažování
- **Kreativní uzly**: Generování textu, obrazů, hudby a kreativních děl
- **Analytické uzly**: Rozpoznávání vzorů, analýza dat a extrakce poznatků
- **Znalostní uzly**: Vyhledávání informací, ověřování a kontextualizace
- **Koordinační uzly**: Podpora spolupráce mezi lidmi a systémy umělé inteligence

### Technická implementace

Síť je implementována prostřednictvím několika vrstev:

- **Výpočetní vrstva**: Využití různorodého hardwaru
- **Paměťová vrstva**: Distribuované ukládání modelů a znalostí
- **Komunikační vrstva**: Efektivní směrování úkolů a výsledků
- **Ověřovací vrstva**: Zajištění kvality a souladu s lidskými hodnotami
- **Řídicí vrstva**: Umožnění kolektivního rozhodování

## 🏗️ Architektura

Architektura Fei je navržena pro rozšiřitelnost a výkon:

```
/
├── fei/                  # Hlavní balík
│   ├── core/             # Základní implementace asistenta
│   │   ├── assistant.py  # Hlavní třída asistenta
│   │   ├── mcp.py        # Integrace služby MCP
│   │   └── task_executor.py # Logika provádění úkolů
│   ├── tools/            # Nástroje pro manipulaci s kódem
│   │   ├── code.py       # Manipulace se soubory a kódem
│   │   ├── registry.py   # Registrace nástrojů
│   │   └── definitions.py # Definice nástrojů
│   ├── ui/               # Uživatelská rozhraní
│   │   ├── cli.py        # Rozhraní příkazového řádku
│   │   └── textual_chat.py # TUI s Textualem
│   └── utils/            # Pomocné moduly
│       ├── config.py     # Správa konfigurace
│       └── logging.py    # Nastavení protokolování
├── memdir_tools/         # Paměťový systém
│   ├── server.py         # HTTP API server
│   ├── memorychain.py    # Distribuovaný paměťový systém
│   └── filter.py         # Motor pro filtrování paměti
└── examples/             # Příklady použití
```

## 📚 Dokumentace

Projekt Fei obsahuje komplexní dokumentaci v adresáři `docs/`:

### Základní dokumenty

- [FEI Manifest](docs/FEI_MANIFESTO.md): Prohlášení digitální nezávislosti a kolektivní inteligence
- [Jak funguje síť FEI](docs/HOW_FEI_NETWORK_WORKS.md): Podrobné vysvětlení distribuované sítě
- [Stav projektu](docs/PROJECT_STATUS.md): Aktuální stav vývoje a plán
- [Mapa repozitáře](docs/REPO_MAP.md): Nástroje pro pochopení struktury kódu
- [README MemDir](docs/MEMDIR_README.md): Dokumentace k paměťovému systému MemDir
- [README MemoryChain](docs/MEMORYCHAIN_README.md): Dokumentace k distribuovanému paměťovému ledgeru

### Dokumentace funkcí

- [Řešení problémů s vyhledáváním Brave](docs/BRAVE_SEARCH_TROUBLESHOOTING.md): Řešení problémů s webovým vyhledáváním
- [Nástroje pro vyhledávání](docs/SEARCH_TOOLS.md): Průvodce možnostmi vyhledávání kódu
- [README Textual](docs/TEXTUAL_README.md): Dokumentace k rozhraní TUI

## ⚠️ Známé problémy a chyby

### Základní problémy

1. **Zpracování chyb**
   - Obecné zacházení s výjimkami v `/root/fei/fei/core/assistant.py` maskuje specifické chyby
   - Pohlcené výjimky v různých komponentách skrývají základní problémy
   - Chybějící řádné kontroly před přístupem k vnořeným atributům

2. **Souběžné podmínky**
   - Ukončení procesu postrádá řádnou synchronizaci
   - Správa procesů na pozadí má potenciální souběžné podmínky
   - Komplexní zpracování smyčky událostí asyncio potřebuje zlepšení

3. **Výkonnostní problémy**
   - Neefektivní porovnávání glob vzorů s velkými kódovými základnami
   - Detekce binárních souborů je pomalá pro velké soubory
   - Využití paměti může být vysoké při zpracování mnoha souborů

### Omezení nástrojů

1. **Nástroj pro úpravy**
   - Vyžaduje jedinečný kontext pro operace vyhledávání/nahrazení
   - Nepodporuje refaktoring více souborů
   - Omezené možnosti validace

2. **Spouštění shellových příkazů**
   - Interaktivní příkazy nejsou plně podporovány
   - Povolování příkazů je restriktivní
   - Potenciál pro zombie procesy

3. **Integrace MCP**
   - Omezené zpracování chyb pro síťové problémy
   - Žádné automatické opětovné připojení pro selhané služby
   - Omezení velikosti odpovědi

### Problémy paměťového systému

1. **Memdir**
   - Žádný mechanismus pro čištění starých pamětí
   - Funkce kopírování paměti není implementována
   - Vytváření složek není rekurzivní

2. **Memorychain**
   - Mechanismus konsensu je zjednodušený
   - Omezená ochrana proti škodlivým uzlům
   - Zatím bez skutečné implementace blockchainu

## 🗺️ Plán projektu

### v0.1 (Aktuální)
- Základní funkce asistenta
- Nástroje pro manipulaci se soubory
- Integrace služby MCP
- Rozhraní příkazového řádku

### v0.2 (Další)
- Vylepšené zpracování chyb
- Vylepšené pokrytí testy
- Optimalizace výkonu
- Zlepšení dokumentace

### v0.3 (Plánované)
- Základní webové rozhraní
- Základy systému pluginů
- Pokročilé generování kódu
- Operace s více soubory

### v1.0 (Budoucí)
- Stabilní API
- Integrace s IDE
- Komplexní dokumentace
- Paměťový systém připravený pro produkci

## 👥 Přispívání

Příspěvky jsou vítány! Postupujte podle těchto kroků:

1. Forkněte repozitář
2. Vytvořte větev pro funkci (`git checkout -b feature/amazing-feature`)
3. Proveďte změny
4. Přidejte testy pro novou funkčnost
5. Ujistěte se, že všechny testy projdou
6. Commitněte své změny (`git commit -m 'Add amazing feature'`)
7. Pushněte do větve (`git push origin feature/amazing-feature`)
8. Otevřete Pull Request

## 📄 Licence

Tento projekt je licencován pod licencí MIT - viz soubor LICENSE pro detaily.

---

<div align="center">
  <p>
    Vytvořeno s ❤️ týmem Fei
  </p>
  <p>
    <a href="https://github.com/david-strejc">GitHub</a> •
    <a href="https://github.com/david-strejc/fei/issues">Problémy</a> •
    <a href="https://github.com/david-strejc/fei/blob/main/README.md">Dokumentace</a>
  </p>
  <p>
    <i>Síť FEI: Inteligence lidu, lidem, pro lid.</i>
  </p>
</div>
