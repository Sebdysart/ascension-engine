import React from "react";
import { Composition } from "remotion";
import { VideoTemplate } from "./VideoTemplate";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="AscensionVideo"
        component={VideoTemplate as React.FC}
        durationInFrames={450}
        fps={30}
        width={1080}
        height={1920}
        defaultProps={{
          hookText: "I CHANGED MY FACE IN 90 DAYS",
          bodyClips: [],
          ctaText: "DISCIPLINE = RESULTS. DROP YOUR GLOW-UP PROGRESS BELOW.",
          colorGrade: "teal_orange",
          cutRateSec: 2.0,
          archetype: "glow_up",
        }}
      />
    </>
  );
};
