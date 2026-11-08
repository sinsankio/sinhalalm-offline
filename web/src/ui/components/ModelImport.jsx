import { useRef } from "react";

export default function ModelImport({ onImport }) {
  const folderRef = useRef(null);
  const filesRef = useRef(null);

  const pick = (event) => {
    const files = [...event.target.files];
    event.target.value = "";
    if (files.length) onImport(files);
  };

  return (
    <div className="model-import">
      <p className="status-detail">
        Already have the model files? Load them from this device instead of downloading.
      </p>
      <div className="model-import-actions">
        <button type="button" className="secondary" onClick={() => folderRef.current?.click()}>
          Load from a folder
        </button>
        <button type="button" className="secondary" onClick={() => filesRef.current?.click()}>
          Choose files
        </button>
      </div>
      <input ref={folderRef} type="file" webkitdirectory="" multiple hidden onChange={pick} />
      <input ref={filesRef} type="file" multiple hidden onChange={pick} />
    </div>
  );
}
