import { useEffect, useState } from "react";
import getReleaseSummary from "@/services/getReleaseSummary";
import ReleaseSummary from "@/types/ReleaseSummary";
import ReleaseIndexEntry from "@/types/ReleaseIndexEntry";
import { ChevronDownIcon } from "@heroicons/react/24/solid";
import getReleaseUrl from "@/utils/getReleaseUrl";

function ReleaseDropdown() {
  const [summary, setSummary] = useState<ReleaseSummary>({
    release: undefined,
    suites: undefined,
    suite: undefined,
    date: "",
  });

  const [currReleaseIndexEntry, setCurrReleaseIndexEntry] = useState<
    ReleaseIndexEntry | undefined
  >();

  useEffect(() => {
    fetch(
      "https://storage.googleapis.com/crfm-helm-public/config/release_index.json",
    )
      .then((response) => response.json())
      .then((data: ReleaseIndexEntry[]) => {
        // set currReleaseIndexEntry to val where releaseIndexEntry.id matches window.RELEASE_INDEX_ID
        if (window.RELEASE_INDEX_ID) {
          const currentEntry = data.find(
            (entry) => entry.id === window.RELEASE_INDEX_ID,
          );
          setCurrReleaseIndexEntry(currentEntry);
          // handles falling back to HELM lite as was previously done in this file
        } else {
          const currentEntry = data.find((entry) => entry.id === "lite");
          setCurrReleaseIndexEntry(currentEntry);
        }
      })
      .catch((error) => {
        console.error("Error fetching JSON:", error);
      });
  }, []);

  function getReleases(): string[] {
    return currReleaseIndexEntry !== undefined &&
      currReleaseIndexEntry.releases !== undefined
      ? currReleaseIndexEntry.releases
      : ["v1.0.0"];
  }

  useEffect(() => {
    const controller = new AbortController();
    async function fetchData() {
      const summ = await getReleaseSummary(controller.signal);
      setSummary(summ);
    }

    void fetchData();
    return () => controller.abort();
  }, []);

  const releases = getReleases();

  const releaseInfo = `Release ${
    summary.release || summary.suite || "unknown"
  } (${summary.date})`;

  if (releases.length <= 1) {
    return <div>{releaseInfo}</div>;
  }

  return (
    <div className="dropdown">
      <div
        tabIndex={0}
        role="button"
        className="normal-case bg-white border-0"
        aria-haspopup="true"
        aria-controls="menu"
      >
        {releaseInfo}{" "}
        <ChevronDownIcon
          fill="black"
          color="black"
          className="inline text w-4 h-4"
        />
      </div>
      <ul
        tabIndex={0}
        className="dropdown-content z-[1] menu p-1 shadow-lg bg-base-100 rounded-box w-max text-base"
        role="menu"
      >
        {releases.map((release) => (
          <li>
            <a
              href={getReleaseUrl(release, currReleaseIndexEntry)}
              className="block"
              role="menuitem"
            >
              {release}
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default ReleaseDropdown;
