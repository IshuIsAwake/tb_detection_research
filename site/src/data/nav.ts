// The site outline, as the sidebar renders it.
//
// This file lists PAGES only. The sub-sections beneath each page are discovered
// from the DOM at runtime (any element carrying `data-section`), so the outline
// can never drift out of sync with the prose — see components/Sidebar.tsx and
// the ReadSection component in components/Reading.tsx.

export interface NavPage {
  path: string;
  /** Section number shown in the outline. Omitted for non-numbered pages. */
  n?: number;
  label: string;
  icon: string;
  /** One line, shown on the home contents list. */
  blurb: string;
}

export interface NavGroup {
  id: string;
  label: string;
  pages: NavPage[];
}

export const NAV_GROUPS: NavGroup[] = [
  {
    id: "review",
    label: "Literature review",
    pages: [
      {
        path: "/about-tb",
        n: 1,
        label: "About TB",
        icon: "activity",
        blurb: "The disease, its two forms, and the scale of the global burden.",
      },
      {
        path: "/how-tb-detected",
        n: 2,
        label: "How TB is detected",
        icon: "search",
        blurb: "Why we screen rather than treat everyone, and how latent and active TB are found.",
      },
      {
        path: "/active-tb-detection",
        n: 3,
        label: "Detecting active TB",
        icon: "stethoscope",
        blurb: "The four frontline diagnostic methods, compared.",
      },
      {
        path: "/history-cad",
        n: 4,
        label: "A history of TB CAD",
        icon: "clock",
        blurb: "Seventy years of teaching computers to read a chest X-ray — and the two times the field changed its mind.",
      },
      {
        path: "/architectures",
        n: 5,
        label: "Deep learning architectures",
        icon: "layers",
        blurb: "CNNs, ResNets, U-Nets, detectors, Transformers and Mamba — what each one does and why it matters here.",
      },
    ],
  },
  {
    id: "log",
    label: "Research log",
    pages: [
      {
        path: "/results",
        label: "Results & ablation",
        icon: "flask-conical",
        blurb: "Every experiment run, what it settled, and what it retracted.",
      },
      {
        path: "/data",
        label: "Datasets",
        icon: "database",
        blurb: "The eight candidate TB datasets and why we train on TBX11K.",
      },
      {
        path: "/references",
        label: "Reading list",
        icon: "book-open",
        blurb: "The papers this project is built on, in dependency order.",
      },
    ],
  },
];

/** Flat list of every page, in outline order. */
export const NAV_PAGES: NavPage[] = NAV_GROUPS.flatMap((g) => g.pages);

/** The literature-review pages, i.e. the "paper" itself. */
export const REVIEW_PAGES: NavPage[] = NAV_GROUPS[0].pages;

export const pageFor = (path: string): NavPage | undefined =>
  NAV_PAGES.find((p) => p.path === path);

/** Previous / next within the literature review, for the page footer pager. */
export function reviewNeighbours(path: string): { prev?: NavPage; next?: NavPage } {
  const i = REVIEW_PAGES.findIndex((p) => p.path === path);
  if (i === -1) return {};
  return { prev: REVIEW_PAGES[i - 1], next: REVIEW_PAGES[i + 1] };
}
