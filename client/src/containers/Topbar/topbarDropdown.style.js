import styled from 'styled-components';
import { palette } from 'styled-theme';
import { transition, borderRadius, boxShadow } from '../../settings/style-util';
import Popover from '../../components/uielements/popover';
import WithDirection from '../../settings/withDirection';
import IconButton from '../../components/uielements/iconbutton';
import Icons from '../../components/uielements/icon';

const TopbarDropdownWrapper = styled(Popover)`
  z-index: 1401;

  &.userPopover {
    margin-top: 50px;
  }

  .scrollbar-track-x {
    opacity: 0 !important;
  }
`;

const IconButtons = styled(IconButton)`
  width: auto;
  height: auto;
  color: #ffffff;

  .userImgWrapper {
    width: 35px;
    height: 35px;
    display: inline-block;
    overflow: hidden;
    border-radius: 50%;

    img {
      max-width: 100%;
    }
  }
`;

const Icon = styled(Icons)`
  font-size: 21px;
  color: ${palette('grey', 6)};
  margin-right: 25px;
  ${transition()};
`;

const TopbarDropdown = styled.div`
  display: flex;
  flex-direction: column;
  background-color: #ffffff;
  margin: 0;
  width: 290px;
  height: 260px;
  min-width: 160px;
  flex-shrink: 0;
  ${borderRadius('2px')};
  ${boxShadow('0 2px 4px rgba(0,0,0,0.26)')};
  ${transition()};
  box-sizing: border-box;
  cursor: default;

  * {
    box-sizing: border-box;
  }
`;

const UserInformation = styled.div`
  width: 100%;
  display: flex;
  padding: 15px 20px;
  background-color: ${palette('grey', 1)};
  border-bottom: 1px solid ${palette('grey', 3)};

  .userImage {
    width: 40px;
    height: 40px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    overflow: hidden;
    border-radius: 50%;

    img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }
  }

  .userDetails {
    width: 100%;
    padding-left: 20px;
    flex-direction: column;

    h3 {
      font-size: 18px;
      color: ${palette('grey', 8)};
      margin: 0 0 5px;
      font-weight: 500;
    }

    p {
      font-size: 12px;
      color: ${palette('grey', 8)};
      margin: 0;
      font-weight: 500;
    }
  }
`;

const SettingsList = styled.div`
  display: flex;
  flex-direction: column;
  padding: 10px 0;

  .dropdownLink {
    font-size: 14px;
    color: ${palette('grey', 8)};
    line-height: 1.1;
    padding: 0;
    background-color: transparent;
    text-decoration: none;
    display: flex;
    align-items: center;
    justify-content: flex-start;
    cursor: pointer;
    margin-bottom: 0;
    padding: 10px 30px;
    background-color: #fff;
    ${transition()};

    &:last-child {
      margin-bottom: 0;
    }

    &:hover {
      background-color: ${palette('grey', 1)};
    }
  }
`;

export { TopbarDropdown, IconButtons, UserInformation, SettingsList, Icon };

export default WithDirection(TopbarDropdownWrapper);
