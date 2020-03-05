import React from 'react';
import { SingleContactCard, Avatar } from './contact.style';

export default ({ contact, setSelectedContact }) => {
  return (
    <SingleContactCard onClick={() => setSelectedContact(contact)}>
      <Avatar alt={contact.name} src={contact.avatar} />
      <h2>{contact.name || 'No Name'}</h2>
    </SingleContactCard>
  );
};
