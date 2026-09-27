// Voice personas — display-only (D1 pin). Backend (providers.py) is voice-truth:
// PERSONA_REGIONS + VOICE_RATE; server resolves selected_persona -> region.
// No voice URLs/models here. Avatars are inline SVG strings (D4 pin), no files.
export interface Persona {
  id: string;
  region: "Bac" | "Trung" | "Nam";
  name: string;
  avatarSvg: string;
  greeting: string;
}

const FACE = (bg: string, skin: string, hair: string, extra: string) =>
  `<svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" role="img" aria-hidden="true">` +
  `<circle cx="32" cy="32" r="30" fill="${bg}"/>` +
  `<circle cx="32" cy="34" r="16" fill="${skin}"/>` +
  `<path d="M16 30c0-10 7-16 16-16s16 6 16 16c0 2-1 3-2 3s-1-4-3-6c-4 2-8 3-11 3s-7-1-11-3c-2 2-2 6-3 6s-2-1-2-3z" fill="${hair}"/>` +
  extra +
  `<circle cx="26" cy="34" r="2" fill="#1e293b"/><circle cx="38" cy="34" r="2" fill="#1e293b"/>` +
  `<path d="M26 42c2 3 10 3 12 0" stroke="#1e293b" stroke-width="2" fill="none" stroke-linecap="round"/>` +
  `</svg>`;

export const PERSONAS: Persona[] = [
  {
    id: "lan",
    region: "Bac",
    name: "Cô Lan",
    avatarSvg: FACE("#dbeafe", "#fde6c8", "#3b2f2f", `<circle cx="32" cy="52" r="4" fill="#2563eb"/>`),
    greeting:
      "Chào bác, con là Lan. Từ hôm nay, con sẽ đồng hành cùng bác theo dõi đường huyết mỗi ngày nhé.",
  },
  {
    id: "huong",
    region: "Trung",
    name: "Cô Hương",
    avatarSvg: FACE("#fef3c7", "#fbd9a8", "#5b3a1e", `<circle cx="32" cy="52" r="4" fill="#d97706"/>`),
    greeting:
      "Chào bác, con là Hương. Bác cần chi, cứ nói với con, con giúp bác ghi lại chỉ số nghe.",
  },
  {
    id: "sau",
    region: "Nam",
    name: "Chú Sáu",
    avatarSvg: FACE("#d1fae5", "#f5cf9f", "#222222", `<rect x="24" y="50" width="16" height="4" rx="2" fill="#059669"/>`),
    greeting:
      "Chào bác, con là Sáu. Mỗi ngày bác đọc số đo cho con nghe, con ghi lại cho bác nghen.",
  },
];

export const DEFAULT_PERSONA_ID = "lan";

export function getPersona(id?: string | null): Persona {
  const found = PERSONAS.find((p) => p.id === (id || "").trim().toLowerCase());
  return found || PERSONAS[0];
}
