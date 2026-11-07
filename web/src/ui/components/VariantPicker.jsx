import { useEffect, useId, useRef, useState } from "react";
import { VARIANTS } from "../../worker/variants.js";

const BARS = 4;

function StrengthIcon({ level }) {
  return (
    <svg className="strength" viewBox="0 0 20 16" aria-hidden="true">
      {Array.from({ length: BARS }, (_, index) => (
        <rect
          key={index}
          x={index * 5}
          y={12 - index * 4}
          width="3.5"
          height={4 + index * 4}
          rx="1"
          className={index < level ? "strength-on" : "strength-off"}
        />
      ))}
    </svg>
  );
}

function Option({ dtype, cached }) {
  const variant = VARIANTS[dtype];
  return (
    <span className="variant-option">
      <StrengthIcon level={variant.strength} />
      <span className="variant-text">
        <span className="variant-name">
          {variant.label} <span className="variant-size">{variant.downloadGb} GB</span>
        </span>
        <span className="variant-summary">
          {variant.summary}
          {cached ? " · downloaded" : ""}
        </span>
      </span>
    </span>
  );
}

export default function VariantPicker({ value, cached = {}, disabled, onChange }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(value);
  const root = useRef(null);
  const listId = useId();
  const options = Object.keys(VARIANTS);

  useEffect(() => {
    if (!open) return undefined;
    const close = (event) => {
      if (!root.current?.contains(event.target)) setOpen(false);
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [open]);

  const choose = (dtype) => {
    setOpen(false);
    if (dtype !== value) onChange(dtype);
  };

  const openList = () => {
    setActive(value);
    setOpen(true);
  };

  const onKeyDown = (event) => {
    const index = options.indexOf(active);
    if (!open && ["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
      event.preventDefault();
      openList();
    } else if (open && event.key === "ArrowDown") {
      event.preventDefault();
      setActive(options[Math.min(index + 1, options.length - 1)]);
    } else if (open && event.key === "ArrowUp") {
      event.preventDefault();
      setActive(options[Math.max(index - 1, 0)]);
    } else if (open && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      choose(active);
    } else if (open && event.key === "Escape") {
      setOpen(false);
    }
  };

  return (
    <div className="variant-picker" ref={root}>
      <button
        type="button"
        className="variant-button"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listId}
        aria-label={`Model variant: ${VARIANTS[value].label}`}
        aria-activedescendant={open ? `${listId}-${active}` : undefined}
        disabled={disabled}
        onClick={() => (open ? setOpen(false) : openList())}
        onKeyDown={onKeyDown}
      >
        <Option dtype={value} cached={cached[value]} />
        <span className="variant-caret" aria-hidden="true">
          ▾
        </span>
      </button>
      {open && (
        <ul className="variant-menu" role="listbox" id={listId} aria-label="Model variant">
          {options.map((dtype) => (
            <li
              key={dtype}
              id={`${listId}-${dtype}`}
              role="option"
              aria-selected={dtype === value}
              className={dtype === active ? "variant-item active" : "variant-item"}
              onPointerEnter={() => setActive(dtype)}
              onClick={() => choose(dtype)}
            >
              <Option dtype={dtype} cached={cached[dtype]} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
