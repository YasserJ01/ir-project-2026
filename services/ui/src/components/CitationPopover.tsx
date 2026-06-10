import { useState } from "react";

interface Props {
  num: string;
  docId: string;
}

export default function CitationPopover({ num, docId }: Props) {
  const [show, setShow] = useState(false);

  return (
    <sup
      className="relative cursor-pointer text-indigo-600 hover:text-indigo-800 dark:text-indigo-400 dark:hover:text-indigo-200"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
      onClick={() => setShow((s) => !s)}
    >
      [{num}]
      {show && (
        <span className="absolute bottom-full left-1/2 z-10 mb-1 -translate-x-1/2 whitespace-nowrap rounded-md border border-slate-200 bg-white px-2 py-1 text-xs shadow-md dark:border-slate-600 dark:bg-slate-800 dark:text-slate-100">
          {docId}
        </span>
      )}
    </sup>
  );
}
