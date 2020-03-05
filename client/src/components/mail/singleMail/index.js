import React from 'react';
import { timeDifference } from '../../../helpers/utility';
import { getBucketColor } from '../../../containers/Mail/desktopView';
import ActionButton from './actionButton';

import SingleMailWrapper, {
  SingleMailContents,
  SingleMailHeader,
  MailInfo,
  MailBody,
  SingleMailReply,
  MailOtherAction,
  AddressBox,
  ComposeMail,
  Avatar,
  Icon,
  IconButton,
} from './singleMail.style';

export default ({
  mails,
  currentUser,
  replyMail,
  changeReplyMail,
  selectMail,
  toggleListVisible,
  changeComposeMail,
  selectedMail,
  selectedBucket,
  toggleAddress,
  activeAddress,
  bulkActions,
}) => {
  if (!selectedMail || selectedMail === -1) {
    return <div />;
  }
  const index = mails.findIndex(mail => mail.id === selectedMail);
  const mail = mails[index];
  const recpName = mail.name;
  const signature = {
    splitLet: recpName
      .match(/\b(\w)/g)
      .join('')
      .split('', 2),
  };

  const toggleMailAddress = activeAddress ? 'v1siBl3' : '';
  const toggleIcon = activeAddress ? 'expand_more' : 'expand_less';
  const paddingTop = activeAddress ? '20px' : '0';

  const bucketColor = getBucketColor(mail);

  return (
    <SingleMailWrapper>
      <SingleMailContents>
        <SingleMailHeader>
          <Avatar style={{ backgroundColor: bucketColor }}>
            {mail.img ? (
              <img alt="#" src={mail.img} />
            ) : (
              <span>{signature.splitLet}</span>
            )}
          </Avatar>

          <MailInfo>
            <AddressBox className={toggleMailAddress}>
              <h3>
                {mail.name}{' '}
                <span className="address">&lt;{mail.email}&gt;</span>
              </h3>
              <p className="h1De-m0b">
                to <span className="selfName">me</span>
                <span className="selfEmail">
                  {currentUser.name} &lt;{currentUser.email}&gt;
                </span>
              </p>

              <p
                className="sH0w-m0b"
                onClick={toggleAddress}
                style={{
                  marginTop: activeAddress ? '15px' : 0,
                  marginLeft: activeAddress ? '-33px' : 0,
                }}
              >
                to <span className="selfName">me</span>
                <span className="selfEmail">
                  {currentUser.name}{' '}
                  <span className="toMail">{currentUser.email}</span>
                </span>
              </p>
            </AddressBox>

            <IconButton onClick={toggleAddress}>
              <Icon>{toggleIcon}</Icon>
            </IconButton>
          </MailInfo>

          <MailOtherAction>
            <span className="mailDate">{timeDifference(mail.date)}</span>
            <ActionButton action={bulkActions} />
          </MailOtherAction>
        </SingleMailHeader>

        <MailBody style={{ paddingTop: paddingTop }}>
          <p>{mail.body}</p>
        </MailBody>
      </SingleMailContents>

      <SingleMailReply>
        {replyMail ? (
          <div className="dRaf7-wRapP3r">
            <div className="r3plY-wRapP3r">
              <div className="imgWrapper">
                <img src={currentUser.image} alt={currentUser.name} />
              </div>
              <ActionButton action={bulkActions} draftButton />

              <IconButton
                onClick={event => {
                  changeComposeMail(true);
                }}
              >
                <Icon>exit_to_app</Icon>
              </IconButton>
            </div>

            <ComposeMail
              allMail={mails}
              placeholder=" "
              changeComposeMail={changeComposeMail}
              singleMail
            />
          </div>
        ) : (
          <div
            className="r3plY-wRapP3r"
            onClick={event => changeReplyMail(true)}
          >
            <div className="imgWrapper">
              <img src={currentUser.image} alt={currentUser.name} />
            </div>
            <span className="replyBtn">Reply</span>

            <IconButton
              onClick={event => {
                event.stopPropagation();
                changeComposeMail(true);
              }}
            >
              <Icon>forward</Icon>
            </IconButton>
          </div>
        )}
      </SingleMailReply>
    </SingleMailWrapper>
  );
};
