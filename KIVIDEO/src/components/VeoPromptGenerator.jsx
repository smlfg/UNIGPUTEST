import React, { useState } from 'react';
import { Copy, Download, ChevronDown, ChevronUp } from 'lucide-react';

const VideoSegment = ({ segment, index }) => {
  const [expanded, setExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="border rounded-lg p-4 mb-4 bg-white shadow-sm">
      <div
        className="flex justify-between items-center cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex-1">
          <h3 className="font-bold text-lg">
            Segment {index + 1}: {segment.timespan}
          </h3>
          <p className="text-sm text-gray-600">{segment.description}</p>
        </div>
        {expanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
      </div>

      {expanded && (
        <div className="mt-4 space-y-3">
          <div className="bg-gray-50 p-3 rounded">
            <p className="text-xs text-gray-500 mb-1">Technische Details:</p>
            <p className="text-sm">{segment.technical}</p>
          </div>

          <div className="bg-blue-50 p-4 rounded relative">
            <div className="flex justify-between items-start mb-2">
              <p className="text-xs font-semibold text-blue-700">
                VEO 3.1 PROMPT:
              </p>
              <button
                onClick={() => copyToClipboard(segment.prompt)}
                className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800"
              >
                <Copy size={14} />
                {copied ? 'Kopiert!' : 'Kopieren'}
              </button>
            </div>
            <p className="text-sm leading-relaxed whitespace-pre-line">
              {segment.prompt}
            </p>
          </div>

          <div className="bg-yellow-50 p-3 rounded">
            <p className="text-xs text-yellow-700 font-semibold mb-1">
              Kontinuität beachten:
            </p>
            <p className="text-sm">{segment.continuity}</p>
          </div>
        </div>
      )}
    </div>
  );
};

const VeoPromptGenerator = () => {
  const segments = [
    {
      timespan: '00:00 – 00:08',
      description: 'Intro mit Text-Overlay und Sprecher',
      technical: 'Statische Halbtotale, helles Tageslicht, leicht bedeckt',
      prompt: `Medium shot of a man in blue shirt holding a document (building voucher) in a sunny outdoor setting, centered composition, natural daylight with slight cloud cover, professional yet informal atmosphere. Text overlay appears: "Mauer Apple Tree Competition 2025 Bündnis 90 die Grünen". Static camera, documentary style, 4K quality, natural colors, shallow depth of field with background slightly blurred.`,
      continuity:
        'Etabliere den visuellen Stil: helles Tageslicht, natürliche Farben, dokumentarischer Look',
    },
    {
      timespan: '00:08 – 00:16',
      description: 'Apfelbaum-Präsentation in Baumschule',
      technical: 'Statische Halbtotale, heller sonniger Tag',
      prompt: `Medium shot of a young apple tree sapling being held up by hands in a tree nursery, bright sunny daylight, outdoor garden center setting with plants in background, the tree fills most of the frame, informative atmosphere. Text overlay lists required items. Static camera, documentary style, natural lighting, crisp focus on the tree.`,
      continuity:
        'Gleiche Lichtverhältnisse wie Segment 1, Außenbereich bleibt konsistent',
    },
    {
      timespan: '00:16 – 00:24',
      description: 'Detail-Aufnahme Apfelsorte',
      technical: 'Halbtotale mit subtiler Schwenkbewegung',
      prompt: `Close-up to medium shot of an apple tree variety labeled "Goldperme", focusing on trunk and crown, subtle slow tilt camera movement from bottom to top showing the entire tree, bright daylight, educational atmosphere. Text overlay explains variety selection. Natural outdoor lighting, sharp focus, documentary style.`,
      continuity:
        'Gleicher Baum wie vorheriges Segment, achte auf Stamm-Konsistenz',
    },
    {
      timespan: '00:24 – 00:32',
      description: 'Sprecher wendet sich an Kamera',
      technical: 'Statische Nahaufnahme/Halbtotale',
      prompt: `Medium close-up of man in blue shirt speaking directly to camera, holding apple tree sapling, bright daylight with direct lighting, personal and engaging atmosphere, outdoor setting, static camera, natural colors, professional documentary style, shallow depth of field.`,
      continuity:
        'Gleicher Protagonist wie Segment 1, blaues Hemd, gleicher Baum',
    },
    {
      timespan: '00:32 – 00:40',
      description: 'Baum wird hochgehalten',
      technical: 'Statische Halbtotale, natürliches Licht',
      prompt: `Medium shot of man holding apple tree sapling up high, bright natural daylight, outdoor setting, focused on man and tree trunk, static camera, no text overlays, clean composition, documentary style, 4K quality.`,
      continuity:
        'Gleiche Person und Baum, Übergang von Indoor zu Outdoor vorbereiten',
    },
    {
      timespan: '00:40 – 00:48',
      description: 'Ackerland-Panorama',
      technical: 'Halbtotale/Totale mit Schwenk rechts nach links',
      prompt: `Wide shot of fertile farmland field, camera pans slowly from right to left revealing the expansive plot of land, sunny day with bright lighting, active and ready-to-work atmosphere, spade already visible in use, text overlay describes the scene. Smooth camera movement, landscape cinematography, natural colors.`,
      continuity:
        'Umgebungswechsel: von Baumschule zu Feld. Etabliere neue Location',
    },
    {
      timespan: '00:48 – 00:56',
      description: 'Detail: Spaten und Pflanzanleitung',
      technical: 'Nahaufnahme, statische Kamera',
      prompt: `Close-up shot of spade stuck in soil, then cut to extreme close-up of planting instruction paper, bright daylight, focused detail-oriented atmosphere, static camera, shallow depth of field, documentary style. Sharp focus on objects, natural earth tones.`,
      continuity:
        'Gleiche Feld-Location, Detailaufnahmen als Vorbereitung',
    },
    {
      timespan: '00:56 – 01:04',
      description: 'Mann nähert sich Pflanzloch',
      technical: 'Halbtotale mit leichter Kamerabewegung',
      prompt: `Medium shot of a clearly visible planting hole in ground, man approaches and steps toward the hole, slight tracking camera movement following his motion, bright daylight, atmosphere of progress, text overlay present, documentary style, natural colors.`,
      continuity: 'Gleiche Feld-Location, Protagonist kommt ins Bild',
    },
    {
      timespan: '01:04 – 01:12',
      description: 'Baum wird eingepflanzt',
      technical: 'Halbtotale/Totale mit Schwenkbewegung',
      prompt: `Medium to wide shot of apple tree sapling being placed into prepared hole in ground, slight pan camera movement following the tree placement action, bright daylight, focus on main task of tree planting, documentary style, smooth motion, natural earth colors.`,
      continuity:
        'Gleicher Baum wie vorherige Segmente, gleiche Location',
    },
    {
      timespan: '01:12 – 01:20',
      description: 'Grasbüschel für Gießrand',
      technical: 'Nahaufnahme, statisch',
      prompt: `Close-up shot of grass clumps for watering edge lying on ground, static camera, bright daylight, detail focus atmosphere, materials positioned on soil, text overlay describes element, sharp focus, natural colors, documentary style.`,
      continuity:
        'Detail-Insert, gleiche Feld-Location und Lichtverhältnisse',
    },
    {
      timespan: '01:20 – 01:28',
      description: 'Rasenfreie Baumscheibe',
      technical: 'Halbtotale, statisch',
      prompt: `Medium shot of lawn-free tree disc (round fleece or mulch) laid ready on ground, man standing beside it, bright daylight with clear visibility, static camera, text overlay present, documentary style, natural colors, overhead-angled composition.`,
      continuity: 'Gleicher Protagonist, gleiche Location',
    },
    {
      timespan: '01:28 – 01:36',
      description: 'Gießkanne Detail',
      technical: 'Nahaufnahme, statisch',
      prompt: `Close-up shot of watering can filled with water ready for use, static camera, bright daylight, detail focus on object, text overlay describes item, green/metallic watering can, sharp focus, natural lighting, documentary style.`,
      continuity:
        'Detail-Insert, Vorbereitung für nächste Aktion',
    },
    {
      timespan: '01:36 – 01:44',
      description: 'Grüner Daumen (humorvoll)',
      technical: 'Nahaufnahme von Mann und Daumen',
      prompt: `Close-up shot of man presenting his thumb (symbol of "green thumb") to camera with slight smile, focus on thumb and face, bright daylight, humorous direct atmosphere, static camera, text overlay confirms concept, documentary style with playful tone.`,
      continuity:
        'Gleicher Protagonist wie vorher, persönliche Ebene',
    },
    {
      timespan: '01:44 – 01:52',
      description: 'Pflanzbottich mit Erde',
      technical: 'Nahaufnahme, statisch',
      prompt: `Close-up shot of planting bucket (pail) filled with soil, static camera, bright daylight, detail focus, text overlay present, natural earth colors, sharp focus on bucket, documentary style.`,
      continuity: 'Detail-Insert, Materialien für Pflanzung',
    },
    {
      timespan: '01:52 – 02:00',
      description: 'Baum wird gegossen',
      technical: 'Halbtotale, Fokus auf Gießvorgang',
      prompt: `Medium shot of apple tree being watered with watering can, water flowing onto soil around tree base, bright daylight, practical implementation atmosphere, focus on watering action, documentary style, natural motion, earth tones.`,
      continuity:
        'Gleicher Baum, gleiche Location, praktische Handlung',
    },
    {
      timespan: '02:00 – 02:08',
      description: 'Zwei Männer verdichten Erde',
      technical: 'Halbtotale, statisch',
      prompt: `Medium shot of two men compacting soil around tree by stamping with feet, teamwork atmosphere, bright daylight, focus on community work, static camera, documentary style, natural colors, both men visible in frame.`,
      continuity: 'Zweite Person hinzugefügt, Teamwork-Dynamik',
    },
    {
      timespan: '02:08 – 02:16',
      description: 'Gießrand wird gesetzt',
      technical: 'Halbtotale, Fokus auf Hände',
      prompt: `Medium shot focusing on hands placing grass clumps to create watering rim around tree, bright daylight, detail of action atmosphere, documentary style, natural earth tones, shallow depth of field on hands.`,
      continuity:
        'Detail-Fokus, praktische Gartenarbeit',
    },
    {
      timespan: '02:16 – 02:24',
      description: 'Fertigstellung Gießrand',
      technical: 'Halbtotale mit leichter Schwenkbewegung',
      prompt: `Medium shot of watering rim being finalized, slight pan camera movement checking the rim's completion, bright daylight, focused completion atmosphere, documentary style, circular composition showing the rim around tree base.`,
      continuity:
        'Gleiche Aktion wie vorheriges Segment, Fortschritt zeigen',
    },
    {
      timespan: '02:24 – 02:32',
      description: 'Mann erklärt Gießrand-Nutzen',
      technical: 'Halbtotale des fertigen Gießrands',
      prompt: `Medium shot of completed watering rim around tree, man explaining its purpose (rainwater retention), bright daylight, explanatory atmosphere, static camera, documentary style, clear view of circular rim structure.`,
      continuity:
        'Gleicher Protagonist, zeigt fertige Arbeit',
    },
    {
      timespan: '02:32 – 02:40',
      description: 'Erde wird verdichtet',
      technical: 'Nahaufnahme der Erde',
      prompt: `Close-up shot of soil being compacted around tree base with hands or feet, bright daylight, tactile detail atmosphere, natural earth colors, documentary style, focus on soil texture.`,
      continuity:
        'Detail-Insert, finale Arbeitsschritte',
    },
    {
      timespan: '02:40 – 02:48',
      description: 'Begutachtung der Pflanzstelle',
      technical: 'Halbtotale, statisch',
      prompt: `Medium shot of finished planting site being inspected, satisfaction atmosphere, man examining the completed work deemed "perfectly fitted", bright daylight, static camera, documentary style, overview of tree and rim.`,
      continuity:
        'Übersicht der fertigen Arbeit, Zufriedenheit zeigen',
    },
    {
      timespan: '02:48 – 02:56',
      description: 'Pflanzpfahl wird eingeschlagen',
      technical: 'Halbtotale, Fokus auf Handlung',
      prompt: `Medium shot of second man (Manfred) hammering plant stake into ground with sledgehammer, active dynamic work atmosphere, bright daylight, focus on action, documentary style, motion captured mid-swing.`,
      continuity:
        'Zweiter Mann wieder aktiv, dynamische Aktion',
    },
    {
      timespan: '02:56 – 03:04',
      description: 'Auftritt Wachstumsspezialist',
      technical: 'Halbtotale mit Schwenkbewegung',
      prompt: `Medium shot of plant stake driven into ground, camera pans slightly to follow new character entering scene - "growth specialist" wearing hat and beard, bright daylight, humorous mood shift atmosphere, documentary style with playful tone.`,
      continuity:
        'Neue Figur einführen: Hut, Bart, theatralische Präsenz',
    },
    {
      timespan: '03:04 – 03:12',
      description: 'Spezialist spricht zur Kamera',
      technical: 'Statische Halbtotale',
      prompt: `Medium shot of bearded man with hat speaking directly to camera with theatrical expression, bright daylight, humorous theatrical atmosphere, static camera, documentary style with comedic element, character positioned center frame.`,
      continuity:
        'Gleicher Charakter, theatralische Performance',
    },
    {
      timespan: '03:12 – 03:20',
      description: 'Wachstumselixier-Flasche',
      technical: 'Nahaufnahme von Flasche und Mann',
      prompt: `Close-up shot of man holding bottle labeled "growth elixir" and small vessel, detail of props, bright daylight, focus on ritual atmosphere, documentary style with theatrical element, sharp focus on bottle.`,
      continuity:
        'Gleicher Charakter mit Hut, Requisit einführen',
    },
    {
      timespan: '03:20 – 03:28',
      description: 'Elixier wird gegossen',
      technical: 'Nahaufnahme der Handlung',
      prompt: `Close-up shot of liquid being poured from bottle into soil around tree base, ritual focus atmosphere, bright daylight, documentary style with magical realism element, focus on pouring action.`,
      continuity:
        'Gleiche Flasche und Hand, rituelle Handlung',
    },
    {
      timespan: '03:28 – 03:36',
      description: 'Zaubergeste',
      technical: 'Halbtotale mit Fokus auf Geste',
      prompt: `Medium shot of bearded man in hat making magical gesture with hands, speaking spell words "im Salbaum", theatrical staging atmosphere, bright daylight, documentary style with fantasy element, dynamic hand movement.`,
      continuity:
        'Gleicher Charakter, theatralische Performance fortsetzen',
    },
    {
      timespan: '03:36 – 03:44',
      description: 'Zauberspruch fortgesetzt',
      technical: 'Halbtotale, Fokus auf Mann',
      prompt: `Medium shot of man continuing spell incantation "Wachs in großen Raum Wachs wachs", theatrical gestures, bright daylight, humorous magical atmosphere, documentary style with comedic fantasy element.`,
      continuity:
        'Gleicher Charakter, gleiche Performance',
    },
    {
      timespan: '03:44 – 03:52',
      description: 'Reaktion auf Zauber',
      technical: 'Halbtotale mit Fokus auf ersten Mann',
      prompt: `Medium shot of first man (in blue shirt) turning around with surprised expression saying "Wow ein Superzauber ist gelungen", atmosphere of surprise and humorous reveal, bright daylight, documentary style with comedic timing.`,
      continuity:
        'Zurück zum ersten Protagonisten, Überraschung zeigen',
    },
    {
      timespan: '03:52 – 04:00',
      description: 'Gewachsener Baum erscheint',
      technical: 'Halbtotale von Baum und Männern',
      prompt: `Medium shot of significantly larger grown apple tree standing where sapling was planted, two men standing beside it, bright daylight, final staging atmosphere, documentary style with magical realism, tree clearly taller and more mature.`,
      continuity:
        'WICHTIG: Größerer Baum ersetzt Setzling, gleiche Location, dramatischer Unterschied',
    },
    {
      timespan: '04:00 – 04:08',
      description: 'Mann umarmt Baum',
      technical: 'Halbtotale, statisch',
      prompt: `Medium shot of man in blue shirt hugging tree trunk while speaking to camera, emotional closing atmosphere, bright daylight, static camera, documentary style, heartfelt moment, connection between man and nature.`,
      continuity:
        'Erster Protagonist mit gewachsenem Baum, emotionale Bindung',
    },
    {
      timespan: '04:08 – 04:16',
      description: 'Klimawandel-Botschaft',
      technical: 'Halbtotale, statisch',
      prompt: `Medium shot of man delivering closing message about climate change, serious yet hopeful atmosphere, bright daylight, static camera, documentary style, direct address to camera, tree visible in background.`,
      continuity:
        'Gleiche Szene, wichtige Botschaft, Ernst und Hoffnung',
    },
    {
      timespan: '04:16 – 04:24',
      description: 'Finale mit beiden Männern',
      technical: 'Halbtotale',
      prompt: `Medium shot of both men together, growth specialist stepping into frame, saying "Tada" with cheerful closing atmosphere, bright daylight, documentary style with lighthearted ending, both characters visible, tree in background.`,
      continuity:
        'Beide Charaktere zusammen, fröhlicher Abschluss',
    },
  ];

  const exportAllPrompts = () => {
    const allPrompts = segments
      .map(
        (seg, i) => `SEGMENT ${i + 1} (${seg.timespan})\n${seg.prompt}\n\n`,
      )
      .join('');

    const blob = new Blob([allPrompts], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'veo-prompts-mauer-apple-tree.txt';
    a.click();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-6">
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-xl shadow-lg p-6 mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            🌳 Veo 3.1 Prompt Generator
          </h1>
          <h2 className="text-xl text-gray-600 mb-4">
            Mauer Apple Tree Competition 2025
          </h2>

          <div className="bg-blue-50 border-l-4 border-blue-500 p-4 mb-4">
            <p className="text-sm text-gray-700">
              <strong>Video-Länge:</strong> 4:26 Minuten = 30 Segmente à ~8
              Sekunden
              <br />
              <strong>Stil:</strong> Dokumentarisch, helles Tageslicht,
              informativ-humorvoll
              <br />
              <strong>Hinweis:</strong> Jedes Segment einzeln in Veo 3.1
              generieren, auf Kontinuität achten
            </p>
          </div>

          <button
            onClick={exportAllPrompts}
            className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-6 rounded-lg flex items-center justify-center gap-2 transition-colors"
          >
            <Download size={20} />
            Alle Prompts als TXT exportieren
          </button>
        </div>

        <div className="space-y-2">
          {segments.map((segment, index) => (
            <VideoSegment key={index} segment={segment} index={index} />
          ))}
        </div>

        <div className="bg-yellow-50 border-l-4 border-yellow-500 p-4 mt-6 rounded">
          <h3 className="font-bold text-yellow-800 mb-2">
            ⚠️ Wichtige Tipps für Veo 3.1:
          </h3>
          <ul className="text-sm text-yellow-900 space-y-1">
            <li>• Generiere mehrere Varianten pro Segment (3-5x)</li>
            <li>
              • Achte auf Kontinuität: gleicher Protagonist, gleicher Baum,
              gleiche Location
            </li>
            <li>
              • Segment 23-24 (Baum-Wachstum): Hier wird ein GRÖSSERER Baum
              benötigt
            </li>
            <li>
              • Speichere beste Takes und nutze diese als Referenz für folgende
              Segmente
            </li>
            <li>• Text-Overlays müssen in Post-Production hinzugefügt werden</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default VeoPromptGenerator;
