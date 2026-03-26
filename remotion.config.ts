import { Config } from "@remotion/cli/config";

Config.setVideoImageFormat("jpeg");
Config.setOverwriteOutput(true);

// Public folder is where clips are served from
Config.setPublicDir("./public");
