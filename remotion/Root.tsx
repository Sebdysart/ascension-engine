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
          bodyClips: [],
          colorGrade: "dark_cinema",
          zoomPunch: true,
          showOverlay: false,
          musicVolume: 0.85,
        }}
      />
    </>
  );
};
