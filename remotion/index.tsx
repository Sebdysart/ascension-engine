import { Composition, registerRoot } from "remotion";
import { VideoTemplate } from "./VideoTemplate";
import { MogEdit } from "./compositions/MogEdit";
import { BpRating } from "./compositions/BpRating";
import { IronicProvocation } from "./compositions/IronicProvocation";
import { BrutalBeatMontage, SAMPLE_EDL } from "./BrutalBeatMontage";
import React from "react";

const RemotionRoot = () => (
  <>
    <Composition
      id="AscensionVideo"
      component={VideoTemplate}
      durationInFrames={450}
      fps={30}
      width={1080}
      height={1920}
      defaultProps={{
        hookText: "",
        bodyClips: [],
        ctaText: "",
        colorGrade: "dark_cinema",
        cutRateSec: 2.0,
        archetype: "mog_edit",
        showBeforeAfter: false,
      }}
    />
    <Composition
      id="MogEdit"
      component={MogEdit}
      durationInFrames={450}
      fps={30}
      width={1080}
      height={1920}
      defaultProps={{
        clips: [],
        colorGrade: "dark_cinema",
        musicPath: "",
        caption: "",
        watermark: "ASCENSION",
        lyricText: "",
        showLyric: false,
      }}
    />
    <Composition
      id="BpRating"
      component={BpRating}
      durationInFrames={300}
      fps={30}
      width={1080}
      height={1920}
      defaultProps={{
        clipPath: "",
        revealWord: "MOGGED",
        colorGrade: "natural",
        musicPath: "",
        watermark: "ASCENSION",
      }}
    />
    <Composition
      id="IronicProvocation"
      component={IronicProvocation}
      durationInFrames={300}
      fps={30}
      width={1080}
      height={1920}
      defaultProps={{
        clipPath: "",
        line1: "unpopular opinion",
        line2: "I've never seen a pretty blond",
        colorGrade: "dark_indoor",
        musicPath: "",
      }}
    />
    <Composition
      id="BrutalBeatMontage"
      component={BrutalBeatMontage}
      durationInFrames={450}
      fps={30}
      width={1080}
      height={1920}
      defaultProps={{
        edl: SAMPLE_EDL,
        audioPath: "",
      }}
    />
  </>
);

registerRoot(RemotionRoot);
