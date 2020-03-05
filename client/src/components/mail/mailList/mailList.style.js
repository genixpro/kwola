import styled from 'styled-components';
import { palette } from 'styled-theme';
import WithDirection from '../../../settings/withDirection';
import Checkboxes from '../../uielements/checkbox';
import Icons from '../../uielements/icon';
import MailActions from '../singleMailActions';
import ExpansionPanels from '../../uielements/expansionPanel';

const Icon = styled(Icons)``;
const ExpansionPanel = styled(ExpansionPanels)``;

const MailAction = styled(MailActions)`
  display: none;
`;
const Checkbox = styled(Checkboxes)`
  width: 30px;
  height: 30px;

  @media only screen and (max-width: 960px) {
    width: 38px;
    height: 38px;
  }
`;

const Avatar = styled.div`
  width: 30px;
  height: 30px;
  display: flex;
  flex-shrink: 0;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  border-radius: 50%;
  z-index: 1;
  position: absolute;
  top: 0;
  left: 0;

  @media only screen and (max-width: 960px) {
    width: 38px;
    height: 38px;
  }

  ${Icon} {
    display: none;
  }

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
  }

  .sPl1t-l3t {
    font-size: 12px;
    color: #ffffff;
    line-height: 1;
    font-weight: 700;
    text-transform: uppercase;
    margin: 0;
  }
`;

const CheckboxWrapper = styled.div`
  display: inline-flex;
  flex-shrink: 0;
  position: relative;
  margin-right: 20px;

  &:hover {
    ${Avatar} {
      opacity: 0;
      visibility: hidden;
    }

    ${Checkbox} {
      z-index: 2;
    }
  }

  @media only screen and (max-width: 960px) {
    ${Avatar} {
      z-index: 1;
    }

    ${Checkbox} {
      opacity: 0;
      z-index: 2;
    }

    &:hover {
      ${Avatar} {
        opacity: 1;
        visibility: visible;
      }
    }
  }
`;

const MailListInfos = styled.div`
  display: flex;
  width: 100%;
  align-items: center;
  overflow: hidden;

  @media only screen and (max-width: 960px) {
    flex-direction: column;
    align-items: flex-start;
  }

  .name {
    font-size: 13px;
    font-weight: 400;
    flex-shrink: 0;
    color: ${palette('grey', 9)};
    margin: 0;
    padding-right: 10px;
    width: 175px;
    line-height: 1.2;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;

    @media only screen and (max-width: 960px) {
      margin-bottom: 4px;
      display: inline-flex;
      order: 1;
      width: 100%;
    }
  }

  .subDesc {
    padding-right: 15px;
    display: flex;
    overflow: hidden;

    .subject {
      font-size: 13px;
      color: ${palette('grey', 9)};
      margin: 0;
      flex-shrink: 0;
      line-height: 1.2;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;

      &:after {
        content: '—';
        color: ${palette('grey', 6)};
        font-weight: 400;
        margin: 0 5px;
      }

      @media only screen and (max-width: 960px) {
        margin-bottom: 5px;
        font-size: 15px;
        width: 100%;

        &::after {
          content: '';
        }
      }
    }

    .description {
      font-size: 13px;
      color: ${palette('grey', 6)};
      margin: 0;
      line-height: 1.2;
      padding-right: 0;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;

      @media only screen and (max-width: 960px) {
        order: 2;
        width: 100%;
      }
    }
  }

  .mailDate {
    font-size: 12px;
    color: ${palette('grey', 8)};
    font-weight: 500;
    margin: 0;
    line-height: 1.2;
    padding-right: 0;
    flex-shrink: 0;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
  }
`;

const MailListItem = styled.div`
  display: flex;
  width: 100%;
  flex-shrink: 0;
  padding: 10px 0;
  align-items: center;
  position: relative;

  &:last-child {
    padding-right: 0;
  }

  &:hover {
    .singleMailActions {
      display: flex;
    }

    .mailDate {
      display: none;
    }
  }

  @media only screen and (max-width: 960px) {
    padding: 13px 0;
    align-items: flex-start;

    .singleMailActions,
    .mailDate {
      display: none;
    }

    &:hover {
      .singleMailActions {
        display: none;
      }
    }
  }

  &.unread {
    ${MailListInfos} {
      .name,
      .subject {
        font-weight: 700;
      }

      @media only screen and (max-width: 960px) {
        .name,
        .subject {
          font-weight: 500;
        }
      }
    }
  }

  &.s3lecTed {
    ${Avatar} {
      background-color: ${palette('grey', 5)} !important;

      ${Icon} {
        font-size: 22px;
        color: #fff;
        display: inline-flex;
      }

      img {
        display: none;
      }

      .sPl1t-l3t {
        display: none;
      }
    }
  }

  &.eXpaNd3d {
    @media only screen and (max-width: 960px) {
      align-items: center;
    }

    .singleMailActions {
      display: flex;
    }

    ${CheckboxWrapper} {
      display: none;
    }

    .name {
      display: none;
    }

    .subDesc {
      width: 100%;
    }

    .subject {
      font-weight: 300;
      font-size: 18px;

      @media only screen and (max-width: 960px) {
        margin-bottom: 0;
      }

      &:after {
        display: none;
      }
    }

    .description {
      display: none;
    }

    .mailDate {
      display: none;
    }
  }
`;

const MailListWrapper = styled.div`
  width: 100%;
  display: flex;
  flex-direction: column;
  padding: 3px;

  @media only screen and (max-width: 960px) {
    padding: 3px 0;
  }

  &.cH3cKed {
    ${Avatar} {
      opacity: 0;
      visibility: hidden;
    }

    ${Checkbox} {
      z-index: 2;
    }

    @media only screen and (max-width: 960px) {
      ${Avatar} {
        opacity: 1;
        visibility: visible;
        z-index: 1;
      }

      ${Checkbox} {
        opacity: 0;
        z-index: 2;
      }

      &:hover {
        ${Avatar} {
          opacity: 1;
          visibility: visible;
        }
      }
    }
  }

  .expansionPanel {
    &:before {
      display: none;
    }
  }

  .expansionPanelSummary {
    display: flex;
    width: 100%;

    @media only screen and (max-width: 960px) {
      padding: 0 15px;
    }
  }
`;

const SingleMailListBlock = styled.div`
  display: flex;
  flex-direction: column;
  width: 100%;
  margin-bottom: 20px;

  &:last-child {
    margin-bottom: 0;
  }

  .receivingDate {
    font-size: 13px;
    color: ${palette('grey', 7)};
    margin: 0 0 13px 30px;
    flex-shrink: 0;
    line-height: 1.2;
    font-weight: 500;
  }
`;

export {
  SingleMailListBlock,
  MailListItem,
  CheckboxWrapper,
  MailListInfos,
  Avatar,
  Icon,
  Checkbox,
  MailAction,
  ExpansionPanel,
};
export default WithDirection(MailListWrapper);
