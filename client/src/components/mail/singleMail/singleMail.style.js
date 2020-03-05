import styled from 'styled-components';
import { palette } from 'styled-theme';
import Icons from '../../uielements/icon';
import IconButtons from '../../uielements/iconbutton';
import ComposeMails from '../composeMail';
import Lists, {
  ListSubheader as ListSubheaders,
  ListItem as ListItems,
  ListItemText as ListItemTexts,
} from '../../uielements/lists';

const Icon = styled(Icons)`
  font-size: 20px;
  color: ${palette('grey', 7)};
`;

const IconButton = styled(IconButtons)`
  width: 30px;
  height: 30px;
  padding: 0;
  margin-right: 5px;

  &:last-child {
    margin-right: 0;
  }
`;

const List = styled(Lists)`
  max-width: 360px;
  width: 100%;

  &.dropdownList {
    padding-bottom: 0;
  }

  &.mAil-dRopd0Wn {
    padding-top: 0;
    padding-bottom: 0;
  }
`;

const ListSubheader = styled(ListSubheaders)`
  height: 35px;
  display: flex;
  width: 100%;
  align-items: center;
  font-size: 12px;
  color: ${palette('grey', 7)};
  font-weight: 500;
  padding-left: 22px;
`;

const ListItem = styled(ListItems)`
  padding: 8px 20px;

  ${Icon} {
    font-size: 19px;
  }
`;

const ListItemText = styled(ListItemTexts)`
  h3 {
    color: ${palette('grey', 6)};
    font-size: 13px;
    text-transform: capitalize;
    font-weight: 500;
  }
`;

const ListLabel = styled.h3`
  font-size: 15px;
  font-weight: 500;
  color: ${palette('grey', 8)};
  padding: 6px 20px 14px;
  margin: 0;
`;

const Avatar = styled.div`
  width: 40px;
  height: 40px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  border-radius: 50%;
  margin-right: 20px;
  flex-shrink: 0;

  span {
    color: rgba(255, 255, 255, 0.9);
    font-size: 16px;
    font-weight: 500;
    line-height: 1;
    margin: 0;
  }
`;

const SingleMailHeader = styled.div`
  display: flex;
  width: 100%;
  align-items: flex-start;
`;

const AddressBox = styled.div`
  display: flex;
  flex-direction: row;
  align-items: baseline;

  @media only screen and (max-width: 960px) {
    flex-direction: column;
  }

  h3 {
    font-size: 13px;
    color: ${palette('grey', 8)};
    font-weight: 700;
    margin: 0 5px 5px 0;
    line-height: 1.2;

    span {
      display: none;
    }
  }

  p {
    font-size: 13px;
    color: ${palette('grey', 7)};
    font-weight: 400;
    margin: 0;
    line-height: 1.2;

    @media only screen and (max-width: 960px) {
      cursor: pointer;
    }

    &.sH0w-m0b {
      display: none;
    }

    @media only screen and (max-width: 960px) {
      &.h1De-m0b {
        display: none;
      }

      &.sH0w-m0b {
        display: inline-block;
      }
    }

    span {
      margin: 0;
    }

    .selfEmail {
      display: none;
    }
  }

  &.v1siBl3 {
    flex-direction: column;

    h3 {
      span {
        display: inline-flex;

        @media only screen and (max-width: 960px) {
          width: 100%;
          font-weight: 400;
          margin-top: 5px;
          color: ${palette('grey', 7)};
        }
      }
    }

    p {
      text-transform: capitalize;

      .selfName {
        display: none;
      }

      .selfEmail {
        display: inline-flex;
        text-transform: none;
        font-size: 13px;

        @media only screen and (max-width: 960px) {
          padding-left: 15px;
          color: ${palette('grey', 8)};
          cursor: pointer;

          .toMail {
            margin-left: 5px;
            color: ${palette('blue', 5)};
          }
        }
      }
    }
  }
`;

const MailInfo = styled.div`
  width: 100%;
  display: flex;
  align-items: flex-start;
  padding-top: 4px;

  ${IconButton} {
    width: 18px;
    height: 18px;
    padding: 0;
    margin-left: 5px;

    @media only screen and (max-width: 960px) {
      display: none;
    }

    ${Icon} {
      font-size: 16px;
    }
  }
`;

const MailOtherAction = styled.div`
  display: inline-flex;
  align-items: center;
  flex-shrink: 0;
  margin-right: -10px;

  @media only screen and (max-width: 960px) {
    position: absolute;
    right: 15px;
  }

  .mailDate {
    font-size: 14px;
    color: ${palette('grey', 6)};
    margin: 0;
    margin-right: 5px;

    @media only screen and (max-width: 960px) {
      font-size: 13px;
    }
  }
`;

const MailBody = styled.div`
  padding: 0 60px;

  @media only screen and (max-width: 959px) {
    padding: 0 60px !important;
    margin-top: 20px;
  }

  @media only screen and (max-width: 700px) {
    padding: 0 !important;
    margin-top: 25px;
  }

  p {
    font-size: 13px;
    color: ${palette('grey', 8)};
    margin: 0 0 26px;
    line-height: 1.6;

    &:last-child {
      margin-bottom: 0;
    }
  }
`;

const ComposeMail = styled(ComposeMails)`
  padding: 0 55px 20px 85px;

  @media only screen and (max-width: 800px) {
    padding: 0 15px 20px;
  }
`;

const SingleMailReply = styled.div`
  border-top: 1px solid ${palette('grey', 3)};

  .r3plY-wRapP3r {
    display: flex;
    align-items: center;
    cursor: text;
    background-color: ${palette('grey', 1)};
    padding: 14px 24px;

    @media only screen and (max-width: 800px) {
      padding: 14px 15px;
    }

    .imgWrapper {
      width: 40px;
      height: 40px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
      border-radius: 50%;
      margin-right: 20px;
      cursor: default;

      img {
        width: 100%;
        height: 100%;
        object-fit: cover;
      }
    }

    .replyBtn {
      font-size: 14px;
      font-weight: 400;
      color: ${palette('grey', 7)};
      margin: 0;
      cursor: pointer;

      &:hover {
        font-weight: 500;
        color: ${palette('indigo', 5)};
      }
    }

    ${IconButton} {
      margin-left: auto;
      margin-right: -10px;
      width: 35px;
      height: 35px;
      padding: 0;

      ${Icon} {
        color: ${palette('grey', 7)};
      }
    }
  }

  .dRaf7-wRapP3r {
    width: 100%;
    display: flex;
    flex-direction: column;

    .r3plY-wRapP3r {
      background-color: #ffffff;
      padding-bottom: 0;

      .draftLabel {
        font-size: 13px;
        font-weight: 400;
        color: ${palette('grey', 7)};
        margin: 0;
        cursor: pointer;
        display: flex;
        align-items: flex-end;

        span {
          font-weight: 700;
          color: ${palette('red', 8)};
          margin-right: 5px;
        }

        ${Icon} {
          font-size: 14px;
          color: ${palette('grey', 7)};
          margin-left: 5px;
          margin-right: 0;
        }
      }
    }

    .eDit0R-wRapP3r {
      padding: 0;

      .mailComposeEditor {
        .quill {
          .ql-container {
            min-height: auto;

            .ql-editor {
              min-height: auto;
            }
          }

          @media only screen and (min-width: 701px) {
            .ql-toolbar {
              width: calc(100% - 70px);
              padding: 4px 8px;
            }
          }
        }
      }

      .sEnd-bTn {
        left: 0;

        @media only screen and (max-width: 700px) {
          left: auto;
        }

        @media only screen and (min-width: 701px) {
          position: absolute;
          bottom: 0;
        }
      }

      .dEl3Te-bTn {
        right: -38px;

        @media only screen and (max-width: 700px) {
          right: auto;
        }

        @media only screen and (min-width: 701px) {
          position: absolute;
          bottom: 0;
        }

        @media only screen and (max-width: 800px) {
          right: -7px;
        }
      }
    }
  }
`;

const SingleMailContents = styled.div`
  padding: 16px 24px 24px;

  @media only screen and (max-width: 960px) {
    padding: 16px 15px 24px;
  }
`;

const SingleMailWrapper = styled.div`
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  height: 100%;
  width: 100%;
`;

export {
  SingleMailContents,
  SingleMailHeader,
  MailInfo,
  MailBody,
  SingleMailReply,
  ComposeMail,
  MailOtherAction,
  AddressBox,
  Avatar,
  Icon,
  IconButton,
  List,
  ListItem,
  ListSubheader,
  ListItemText,
  ListLabel,
};
export default SingleMailWrapper;
