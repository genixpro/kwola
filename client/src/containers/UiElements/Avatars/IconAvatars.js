import React from "react";
import PropTypes from "prop-types";
import Avatar from "./avatars.style";
import Icon from "../../../components/uielements/icon/index.js";
import { DemoWrapper } from "../../../components/utility/papersheet";

function IconAvatars(props) {
  return (
    <DemoWrapper>
      <Avatar>
        <Icon>folder</Icon>
      </Avatar>
      <Avatar className="pinkAvatar">
        <Icon>pageview</Icon>
      </Avatar>
      <Avatar className="greenAvatar">
        <Icon>assignment</Icon>
      </Avatar>
    </DemoWrapper>
  );
}

IconAvatars.propTypes = {
  classes: PropTypes.object.isRequired
};

export default IconAvatars;
