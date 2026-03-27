import React from "react";
import { Composition } from "remotion";
import { VideoTemplate } from "./VideoTemplate";
import { BrutalBeatMontage } from "./compositions/BrutalBeatMontage";

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
          bodyClips: [],
          colorGrade: "dark_cinema",
          zoomPunch: true,
          showOverlay: false,
          musicVolume: 0.85,
        }}
      />
      <Composition
        id="BrutalBeatMontage"
        component={BrutalBeatMontage as React.FC}
        durationInFrames={450}
        fps={30}
        width={1080}
        height={1920}
        defaultProps={{
          slots: [],
          bpm: 114,
          musicPath: undefined,
          watermark: "",
        }}
      />
    </>
  );
};
